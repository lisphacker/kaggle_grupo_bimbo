sample.data.frame <- function(df, size) {
    df[sample(nrow(df), size),]
}
