library(h5)

read.pandas.hdf <- function(file.path) {
  file <- h5file(file.path, "r")
  data <- file["clients"]
  print(list.groups(file))
  h5close(file)
  data
}

read.hdf("../data/python/clients.h5")
