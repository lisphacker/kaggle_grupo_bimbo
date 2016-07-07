read.train <- function() {
  df <- read.table("../data/original/train.csv", header = TRUE, sep = ",",
                   col.names = c("week.num", "sales.depot.id", "sales.channel.id", "route.id",
                                 "client.id", "product.id",
                                 "sales.unit", "sales", "returns.unit", "returns", "adjusted.demand"))
  df
}
