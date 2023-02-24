#CUDA_VISIBLE_DEVICES=4 nohup julia scripts/train_openfwi.jl & > cond.out 2> cond.err &
using DrWatson
@quickactivate :MultiSourceSummary
import Pkg; Pkg.instantiate()

using PyPlot
using InvertibleNetworks
using Flux
using LinearAlgebra
using Random
using JLD2
using Statistics
using ImageQualityIndexes
using DrWatson 
using Images
using MLUtils
using UNet
using FFTW
using BSON 
using HDF5
using SlimPlotting 

function posterior_sampler(G, y, size_x; device=gpu, num_samples=1, batch_size=16)
    # make samples from posterior for train sample 
	X_forward = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat); #needs to set the proper sizes here

    X_post_train = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
    	ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
   		X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
        	ZX_noise_i,
        	Zy_fixed_train
    		)[1] |> cpu;
	end
	X_post_train
end

function get_cm_l2_ssim(G, X_batch, Y_batch; device=gpu, num_samples=1)
	    num_test = size(Y_batch)[end]
	    l2_total = 0 
	    ssim_total = 0 
	    #get cm for each element in batch
	    for i in 1:num_test
	    	y_i = Y_batch[:,:,:,i:i]
	    	x_i = X_batch[:,:,:,i:i]
	    	X_post_test = posterior_sampler(G, y_i, size(x_i); device=device, num_samples=num_samples, batch_size=batch_size)
	    	X_post_mean_test = mean(X_post_test;dims=4)
	    	ssim_total += assess_ssim(X_post_mean_test[:,:,1,1], x_i[:,:,1,1]|> cpu)
			l2_total   += norm(X_post_mean_test[:,:,1,1]- (x_i[:,:,1,1]|> cpu))^2
		end
	return l2_total / num_test, ssim_total / num_test
end

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size=16)
	num_test = size(Y_batch)[end]
	l2_total = 0 
	logdet_total = 0 
	num_batches = div(num_test, batch_size)
	for i in 1:num_batches
		x_i = X_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 
    	y_i = Y_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 

    	x_i .+= noise_lev_x*randn(Float32, size(x_i)); 
    	y_i .+= noise_lev_y*randn(Float32, size(y_i)); 

    	Zx, Zy, lgdet = G.forward(x_i|> device, y_i|> device) |> cpu;
    	l2_total     += norm(Zx)^2 / (N*batch_size)
		logdet_total += lgdet / N
	end

	return l2_total / (num_batches), logdet_total / (num_batches)
end

sim_name = "cond-simple-openfwi"
plot_path = plotsdir(sim_name)

import DrWatson: _wsave
_wsave(s, fig::Figure) = fig.savefig(s, bbox_inches="tight", dpi=100)

# Training hyperparameters 
device = gpu

lr      = 8f-4
clipnorm_val = 3f0
noise_lev_x = 0.01f0
noise_lev_y = 0.0f0

batch_size   = 8#
n_epochs     = 150
num_post_samples = 64

validation_perc = 0.9018

save_every   = n_epochs
plot_every   = 4

sample_i      = 3
sample_i_test = 1
n_condmean    = 20

Xs_pre = h5open("../openfwi/train_seismic_4_8.h5", "r") do file
    read(file, "velocity")
end;
Ys_pre = h5open("../openfwi/train_seismic_4_8.h5", "r") do file
    read(file, "data")
end;
Xs_pre ./= 1000
n_src = size(Ys_pre)[3]
n_total = size(Xs_pre)[end]
n_c = n_src
n_in = 1

Xs = zeros(Float32,64,64,1,n_total);
Ys = zeros(Float32,64,1024,n_src,n_total);
for i in 1:n_total
	Xs[:,:,1,i] = imresize(Xs_pre[:,:,1,i],(64,64))
	for j in 1:n_src
		Ys[:,:,j,i] = imresize(Ys_pre[:,:,j,i],(64,1024))
	end 
end 
nx, ny = size(Xs)[1:2]
N = nx*ny


Random.seed!(123);
(X_train,Y_train), (X_test,Y_test) = splitobs((Xs,Ys); at=validation_perc, shuffle=true);

n_train = size(X_train)[end]
n_test = size(X_test)[end]

# training indexes 
n_batches      = cld(n_train, batch_size)-1
n_train = batch_size*n_batches
n_batches_test = cld(n_test, batch_size)-1
n_test = batch_size*n_batches_test

# Architecture parametrs
separate= false #this is separate tracks so that the summary network only works in one scale
sum_net = true
h2 = nothing
unet_lev = 4
if sum_net
	h2 = Chain(MaxPool((1,16)),Unet(n_c,n_in,unet_lev))
	trainmode!(h2, true)
	h2 = FluxBlock(h2)|> device
end

# Create conditional network
L = 3
K = 9 #good middle ground
n_hidden = 64
low = 0.5f0

Random.seed!(123);
G = NetworkConditionalGlow(1, n_in, n_hidden,  L, K; summary_net=h2, split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0));
G = G |> device;

# Optimizer
decay = 5f-4
opt = Flux.Optimiser(WeightDecay(decay),ClipNorm(clipnorm_val), ADAM(lr))

# Training logs 
loss   = [];
logdet_train = [];
ssim   = [];
l2_cm  = [];

loss_test   = [];
logdet_test = [];
ssim_test   = [];
l2_cm_test  = [];


for e=1:n_epochs# epoch loop
	idx_e = reshape(randperm(n_train), batch_size, n_batches) 
    for b = 1:n_batches # batch loop
    	@time begin
	        X = X_train[:, :, :, idx_e[:,b]];
	        Y = Y_train[:, :, :, idx_e[:,b]];
	        X .+= noise_lev_x*randn(Float32, size(X));
	        Y .+= noise_lev_y*randn(Float32, size(Y));
	
	        # Forward pass of normalizing flow
	        Zx, Zy, lgdet = G.forward(X|> device, Y|> device)

	        # Loss function is l2 norm 
	        append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
	        append!(logdet_train, -lgdet / N) # logdet is internally normalized by batch size

	        # Set gradients of flow and summary network
	        G.backward(Zx / batch_size, Zx, Zy; C_save = Y |> device)

	        for p in get_params(G) 
	          Flux.update!(opt,p.data,p.grad)
	        end
	        clear_grad!(G)

	        print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	            "; f l2 = ",  loss[end], 
	            "; lgdet = ", logdet_train[end], "; f = ", loss[end] + logdet_train[end], "\n")
	        Base.flush(Base.stdout)
    	end
    end
    
    # get objective mean metrics over testing batch  
    @time l2_test_val, lgdet_test_val  = get_loss(G, X_test, Y_test; device=device, batch_size=batch_size)
    append!(logdet_test, -lgdet_test_val)
    append!(loss_test, l2_test_val)

    # get conditional mean metrics over training batch  
    @time cm_l2_train, cm_ssim_train = get_cm_l2_ssim(G, X_train[:,:,:,1:n_condmean], Y_train[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
    append!(ssim, cm_ssim_train)
    append!(l2_cm, cm_l2_train)

    # get conditional mean metrics over testing batch  
    @time cm_l2_test, cm_ssim_test  = get_cm_l2_ssim(G, X_test[:,:,:,1:n_condmean], Y_test[:,:,:,1:n_condmean]; device=device, num_samples=num_post_samples)
    append!(ssim_test, cm_ssim_test)
    append!(l2_cm_test, cm_l2_test)

    if(mod(e,plot_every)==0) 
	    testmode!(h2, true)
	    k = 1
	    for (latent_test_x, latent_test_y) in [[X_train,Y_train], [X_test, Y_test]]
		    num_cols = 7
		    vmax_error = nothing
		    vmax_std = nothing
		    fig = figure(figsize=(20, 10)); 
		    plot_len = 3
		    for i in 1:plot_len
			    indx = i
			    x = latent_test_x[:,:,:,indx:indx] 
			    y = latent_test_y[:,:,:,indx:indx]

			    fig1 = figure(figsize=(20, 10)); 
			    y_sum = G.summary_net.forward(y|>device)|>cpu

			    y_plot = y[:,:,1,1]'
			    a = quantile(abs.(vec(y_plot)), 98/100)
			    subplot(1,2,1); imshow(y_plot, vmin=-a,vmax=a,aspect=0.1,interpolation="none", cmap="seismic")
				axis("off");  colorbar(fraction=0.046, pad=0.04); title(L"$y$")

				y_plot = y_sum[:,:,1,1]'
			    subplot(1,2,2); imshow(y_plot, interpolation="none", cmap="seismic")
				axis("off");  colorbar(fraction=0.046, pad=0.04); title(L"$h_{\phi}(y)$")

			    tight_layout()
		    	fig_name = @strdict   unet_lev nx sum_net clipnorm_val noise_lev_x n_train e lr n_hidden L K batch_size
		  	    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"summary_train.png"), fig1); close(fig1)

			    # make samples from posterior for train sample 
			   	X_post = posterior_sampler(G,  y, size(x); device=device, num_samples=num_post_samples)
			   	X_post = X_post |> cpu

			    X_post_mean = mean(X_post,dims=4)
			    X_post_std  = std(X_post, dims=4)
			    error_mean = abs.(X_post_mean[:,:,1,1]-x[:,:,1,1])
			    ssim_i = round(assess_ssim(X_post_mean[:,:,1,1], x[:,:,1,1]),digits=2)
			    mse_i = round(mean(error_mean.^2),digits=2)

			    y_plot = y[:,:,1,1]'
			    a = quantile(abs.(vec(y_plot)), 98/100)

			    subplot(plot_len,num_cols,(i-1)*num_cols+1); imshow(y_plot, aspect=0.1,vmin=-a,vmax=a,interpolation="none", cmap="seismic")
				colorbar(fraction=0.046, pad=0.04);title(L"$y$")
			
			    subplot(plot_len,num_cols,(i-1)*num_cols+2); imshow(X_post[:,:,1,1]', vmin=1.5,vmax=4.5, interpolation="none", cmap="cet_rainbow4")
				axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample")

				subplot(plot_len,num_cols,(i-1)*num_cols+3); imshow(X_post[:,:,1,2]', vmin=1.5,vmax=4.5, interpolation="none", cmap="cet_rainbow4")
				axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample")

				x_plot = x[:,:,1,1]
				subplot(plot_len,num_cols,(i-1)*num_cols+4); imshow(x_plot',  vmin=1.5,vmax=4.5,  interpolation="none", cmap="cet_rainbow4")
				axis("off"); title(L"$\mathbf{x_{gt}}$") ; colorbar(fraction=0.046, pad=0.04)

				subplot(plot_len,num_cols,(i-1)*num_cols+5); imshow(X_post_mean[:,:,1,1]' ,  vmin=1.5,vmax=4.5,  interpolation="none", cmap="cet_rainbow4")
				axis("off"); title("Conditional mean SSIM="*string(ssim_i)) ; colorbar(fraction=0.046, pad=0.04)

				subplot(plot_len,num_cols,(i-1)*num_cols+6); imshow(error_mean' , vmin=0,vmax=vmax_error, interpolation="none", cmap="magma")
				axis("off");title("Mean squared error "*string(mse_i)) ; cb = colorbar(fraction=0.046, pad=0.04)
				if i ==1 
					vmax_error = cb.vmax
				end

				subplot(plot_len,num_cols,(i-1)*num_cols+7); imshow(X_post_std[:,:,1,1]' , vmin=0,vmax=vmax_std,interpolation="none", cmap="magma")
				axis("off"); title("Standard deviation") ;cb =colorbar(fraction=0.046, pad=0.04)
				if i ==1 
					vmax_std = cb.vmax
				end
			end
			tight_layout()
		    fig_name = @strdict decay n_c unet_lev nx sum_net clipnorm_val noise_lev_x n_train  e lr n_hidden L K batch_size
		    if k == 1
		    	safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_train.png"), fig); close(fig)
			else
				safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_test.png"), fig); close(fig)
			end
			k += 1
		end
		
	    ############# Training metric logs
		sum_train = loss + logdet_train
		sum_test = loss_test + logdet_test

		fig = figure("training logs ", figsize=(10,12))
		subplot(5,1,1); title("L2 Term: train="*string(loss[end])*" test="*string(loss_test[end]))
		plot(loss, label="train");
		plot(n_batches:n_batches:n_batches*e, loss_test, label="test"); 
		axhline(y=1,color="red",linestyle="--",label="Normal Noise")
		ylim(bottom=0.,top=1.5)
		xlabel("Parameter Update"); legend();

		subplot(5,1,2); title("Logdet Term: train="*string(logdet_train[end])*" test="*string(logdet_test[end]))
		plot(logdet_train);
		plot(n_batches:n_batches:n_batches*e, logdet_test);
		xlabel("Parameter Update") ;

		subplot(5,1,3); title("Total Objective: train="*string(sum_train[end])*" test="*string(sum_test[end]))
		plot(sum_train); 
		plot(n_batches:n_batches:n_batches*e, sum_test); 
		xlabel("Parameter Update") ;

		subplot(5,1,4); title("SSIM train $(ssim[end]) test $(ssim_test[end])")
	    plot(1:n_batches:n_batches*e, ssim); 
	    plot(1:n_batches:n_batches*e, ssim_test); 
	    xlabel("Parameter Update") 

	    subplot(5,1,5); title("l2 train $(l2_cm[end]) test $(l2_cm_test[end])")
	    plot(1:n_batches:n_batches*e, l2_cm); 
	    plot(1:n_batches:n_batches*e, l2_cm_test); 
	    xlabel("Parameter Update") 

		tight_layout()
		fig_name = @strdict decay n_c unet_lev nx  sum_net clipnorm_val noise_lev_x n_train e lr  n_hidden L K batch_size
		safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
	end
	 #save params every 4 epochs
    if(mod(e,save_every)==0) 
         # Saving parameters and logs
    	save_dict = nothing;
    	if sum_net
	     	unet_model = G.summary_net.model|> cpu;
	        G_save = deepcopy(G);
	        reset!(G_save.summary_net); # clear params to not save twice
			Params = get_params(G_save) |> cpu;
			global save_dict = @strdict decay n_c  unet_lev unet_model n_in nx sum_net clipnorm_val n_train e noise_lev_x lr n_hidden L K Params loss logdet_train l2_cm ssim loss_test logdet_test l2_cm_test ssim_test batch_size; 
		else 
	        G_save = deepcopy(G);
			Params = get_params(G_save) |> cpu;
			global save_dict = @strdict decay n_c unet_lev  n_in nx sum_net clipnorm_val n_train e noise_lev_x lr n_hidden L K Params loss logdet_train l2_cm ssim loss_test logdet_test l2_cm_test ssim_test batch_size; 
		end;

		@tagsave(
			joinpath(datadir(), savename(save_dict, "bson"; digits=6)),
			save_dict;
			safe=true
		);
		global G = G |> device;
    end
end