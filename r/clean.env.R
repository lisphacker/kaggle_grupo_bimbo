clean.env <- function() {
  names = ls(envir = globalenv())
  for (name in names) {
    if (!is.data.frame(get(name))) {
      rm(list = c(name), envir = globalenv())
    }
  }
}

clean.env()
