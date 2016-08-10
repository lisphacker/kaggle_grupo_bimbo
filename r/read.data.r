read.train <- function() {
  df <- read.table("../data/original/train.csv", header = TRUE, sep = ",",
                   col.names = c("week.num", "sales.depot.id", "sales.channel.id", "route.id",
                                 "client.id", "product.id",
                                 "sales.unit", "sales", "returns.unit", "returns", "adjusted.demand"),
                   colClasses = c("integer", "factor", "factor", "factor",
                                  "factor", "factor",
                                  "integer", "numeric", "integer", "numeric", "integer"))
  df
}

read.test <- function() {
  df <- read.table("../data/original/test.csv", header = TRUE, sep = ",",
                   col.names = c("id", "week.num", "sales.depot.id", "sales.channel.id",
                                 "route.id", "client.id", "product.id"),
                   colClasses = c("integer", "integer", "factor", "factor",
                                  "factor", "factor", "factor"))
  df
}

read.clients <- function() {
  df <- read.table("../data/original/cliente_tabla.csv", header = TRUE, sep = ",",
                   col.names = c("client.id", "client.name"),
                   colClasses = c("factor", "character"))
  df
}


read.products <- function() {
  df <- read.table("../data/original/producto_tabla.csv", header = TRUE, sep = ",",
                   col.names = c("product.id", "product.name"),
                   colClasses = c("factor", "character"))
  df
}

read.towns.and.states <- function() {
  df <- read.table("../data/original/town_state.csv", header = TRUE, sep = ",",
                   col.names = c("sales.depot.id", "town", "state"),
                   colClasses = c("factor", "factor", "factor"))
  df
}

read.data <- function() {
  training.dataset <<- read.train()
  testing.dataset <<- read.test()
  clients <<- read.clients()
  products <<- read.products()
  towns.and.states <<- read.towns.and.states()
  TRUE
}
