function diff_y(a, b)
    a .= @views b[:, 2:end] .- b[:, 1:end - 1]
    # s = size(a)
    # for j = 1:s[2]
    #     @inbounds a[:, j] = b[:, j+1] - b[:, j]
    # end
end

N = 64
nx = N^2
ny = N
a = ones(Float32, nx, ny - 1)
b = ones(Float32, nx, ny)
c = zeros()

using BenchmarkTools
using CuArrays

@btime diff_y($(CuArray(a)), $(CuArray(b)));
@btime diff_y($a, $b)