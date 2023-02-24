using DrWatson
@quickactivate "RosenTrain"

using InvertibleNetworks
using Flux
using LinearAlgebra
using Statistics
using PyPlot
using ProgressMeter: Progress, next!
using MLUtils
using JLD, JLD2
using DrWatson
using SlimPlotting

plot_path = "plots/training";

function posterior_sampler(G, y, size_x; num_samples=16)
    # make samples from posterior
    Y_repeat = repeat(y |> cpu, 1, 1, 1, num_samples) |> device;
    ZX_noise = randn(Float32, size_x[1:(end-1)]..., num_samples) |> device;
    X_post = G.inverse(ZX_noise, Y_repeat);
end

data = load("time1_0217.jld");

X = data["X_1"];
Y = data["Images_1"];

#making them all 320 would be ideal
fig = figure(figsize=(20,12));
d = (6.25f0, 6.25f0);
obs = Int.(round.(range(1, stop=128, length=36)));   # 9 observed time samples
for i = 1:9
    subplot(3,3,i)
    plot_velocity(X[i,:,:], d; name="", new_fig=false); colorbar();clim(0.1,0.7);
end
suptitle("CO2 saturation [0-1]");
fig_name = @strdict 
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_image.png"), fig); 


fig = figure(figsize=(20,12));
obs = Int.(round.(range(1, stop=128, length=36)));   # 9 observed time samples
for i = 1:9
    subplot(3,3,i)
    plot_simage(Y[i,:,:], d; name="", new_fig=false); colorbar();
end
suptitle("data");
fig_name = @strdict 
safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_data.png"), fig); 


# Training hyperparameters
device = cpu 
lr     = 3f-5
epochs = 15
batch_size = 16
clip_norm = 5f0

# Load in training data
n_total = 128
validation_perc = 0.9

nx = 256; ny = 256;
Xs = zeros(Float32, nx, ny, 1, n_total);
Ys = zeros(Float32, nx, ny, 1, n_total);
for i in 1:n_total
    Xs[:,:,:,i] = X[i,end-255:end, end-255:end];
    Ys[:,:,:,i] = Y[i,end-255:end, 1:256]
end


# Use MLutils to split into training and validation set
XY_train, XY_val = splitobs((Xs, Ys); at=validation_perc, shuffle=true);
train_loader = DataLoader(XY_train, batchsize=batch_size, shuffle=true, partial=false);

# Number of training batches 
n_train = numobs(XY_train)
n_val = numobs(XY_val)
batches = cld(n_train, batch_size)
progress = Progress(epochs*batches);

# Architecture parametrs
chan_x = 1    # not RGB so chan=1
chan_y = 1    # not RGB so chan=1
L = 2         # Number of multiscale levels
K = 10        # Number of Real-NVP layers per multiscale level
n_hidden = 32 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(1, 1, n_hidden,  L, K; split_scales=true) |> device;

# Optimizer
opt = Flux.Optimiser(ClipNorm(clip_norm),ADAM(lr))
# opt = ADAM(lr)

# Training logs 
loss_train = [];
loss_val = [];


for e=1:epochs # epoch loop
    for (X, Y) in train_loader #batch loop

        ZX, ZY, logdet_i = G.forward(X|> device, Y|> device);

        G.backward(ZX / batch_size, ZX, ZY)

        for p in get_params(G) 
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(G) # clear gradients unless you need to accumulate

        #Progress meter
        N = prod(size(X)[1:end-1])
        append!(loss_train, norm(ZX)^2 / (N*batch_size) - logdet_i / N)  # normalize by image size and batch size
        next!(progress; showvalues=[
            (:objective, loss_train[end])])
    end

    # Evaluate network on validation set 
    X = getobs(XY_val[1]) 
    Y = getobs(XY_val[2]) 

    ZX, ZY, logdet_i = G.forward(X|> device, Y|> device);

    N = prod(size(X)[1:end-1])
    append!(loss_val, norm(ZX)^2 / (N*n_val) - logdet_i / N)  # normalize by image size and batch size
end



