using FileIO
# img = load("WashingtonianOctober2019SOL-copy.jpg")
img = load("WashingtonianNovember2019SOL-copy.jpg")

height, width = size(img)

using Colors, Images

gray_img = Gray.(img)

using ImageBinarization
alg = Otsu()
bin_img = binarize(gray_img, alg)

arr = Int8.(bin_img)

sum1 = vec(sum(arr, dims=1))
sum2 = vec(sum(arr, dims=2))



function get_break_index(sum_arr)
    flag = [e > 10 for e in sum_arr]
    e = similar(flag)
    e[1: end-1] = flag[2: end]
    f = e - flag
    return findall(x -> x == 1, f)
end

idx2 = get_break_index(sum1)
idx1 = get_break_index(sum2)

for i in 1:length(idx1)-1
    for j in 1:length(idx2)-1
        begin1, end1 = idx2[i] + 3, idx2[i+1] - 3
        begin2, end2 = idx1[j] + 3, idx1[j+1] - 3
        save("$i-$j.jpg", bin_img[begin1: end1, begin2: end2])
    end
end