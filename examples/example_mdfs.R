library(MDFS)

data <- read.csv("data/madelon.50.csv")
c <- read.table("data/test_contrast_variables.csv")

x <- data[, 1:50]
y <- data[, 51]

contrast.mask <- c(rep.int(F, ncol(x)), rep.int(T, ncol(c)))

result_1d <- ComputeMaxInfoGains(x, y, dimensions = 1, seed = 0, discretizations = 30, contrast_data = c)
result_1d_pv <- ComputePValue(
    c(result_1d$IG, attr(result_1d, "contrast_igs")),
    dimensions = 1, divisions = 1, one.dim.mode = "exp", contrast.mask = contrast.mask)

result_2d <- ComputeMaxInfoGains(x, y, dimensions = 2, seed = 0, discretizations = 30, contrast_data = c)
result_2d_pv <- ComputePValue(
    c(result_2d$IG, attr(result_2d, "contrast_igs")),
    dimensions = 2, divisions = 1, one.dim.mode = "exp", contrast.mask = contrast.mask)
