##################################### Function Defination #################################################


####################### Data Loading & Data Reshaping
load_fl_data <- function(path){
  require(abind)
  load_data <- c()
  folderlist <- list.files(path)
  for(i in 1: length(folderlist)){ 
    filelist <- list.files(paste(path, 
                                 folderlist[i], sep = ""))
    dir <- paste(path, folderlist[i], "\\", 
                 filelist, sep = "")
    for(j in 1: length(dir)){
      temp.data <- as.matrix(read.table(dir[j], header = FALSE, sep = "\t", 
                                        skip = 31, dec = "."))
      temp.data <- temp.data[, -1]
      temp.data <- array(temp.data, c(1, dim(temp.data)[1], dim(temp.data)[2]))
      load_data <- abind(load_data, temp.data, along = 1)
    }
    print(i)
  }
  data_x <- array_reshape(load_data, c(dim(load_data)[1], dim(load_data)[2] * dim(load_data)[3]))
  
  cat <- NULL
  for(i in 1 : length(folderlist)){
    cat <- c(cat, rep(folderlist[i], 
                      length(list.files(paste(path, 
                                              folderlist[i], sep = "")))))
  }
  data_y <- cat
  
  list(data_x, data_y)
}

###################### Category Data Transformation
cat_trans <- function(cat){
  data_y <- c()
  for(i in 1: length(cat))
    data_y[i] <- which(cat[i] == unique(cat))-1
  data_y
}
####################### Data Dimension Reduction
data_dim_reduc <- function(data, train_num, ncomp){
  
  train_data <- data[train_num, ]
  test_data <- data[-train_num, ]
  train_data <- scale(train_data)
  test_data <- scale(test_data)
  
  train_svd <- svd(train_data)
  
  svdd2 <- diag(train_svd$d)[1: ncomp, 1: ncomp]
  svdv2 <- train_svd$v[, 1: ncomp]
  svdu2 <- train_svd$u[, 1: ncomp]
  
  train_x <- svdu2 %*% svdd2
  test_x <- test_data %*% svdv2
  list(trainset = train_x, testset = test_x, load_mtrx = svdv2)
}


####################### GAN Function 

GAN <- function(data_x, data_y, latent_size, batch_size = 50, 
                adam_lr = 0.00005, adam_beta_1 = 0.5, epochs = 100){
  require(keras)
  require(progress)
  
  ncomp <- dim(data_x)[2]
  sample <- sample(1: length(data_y), 0.8 * length(data_y), replace = F)
  train_x <- data_x[sample, ]
  test_x <- data_x[-sample, ]
  train_y <- data_y[sample]
  test_y <- data_y[-sample]
  test_loss <- c()
  train_loss <- c()
  class_num <- length(unique(train_y))
  num_train <- length(train_y)
  num_test <- length(test_y)
  
  build_generator <- function(latent_size){
    ########## Map a pair of (z, L), 
    #########  z is a latent vector while L is a label drawn from p_c to generation space
    ann <- keras_model_sequential()
    ann %>% layer_dense(units = ncomp, input_shape = latent_size, activation = "relu")
    
    ########### To generate Z space
    latent <- layer_input(shape = list(latent_size))
    
    ########### Label Inputs
    drug_class <- layer_input(shape = list(1))
    
    ########## 6 Classes in Input data
    cls <- drug_class %>%
      layer_embedding(input_dim = class_num, output_dim = latent_size, 
                      embeddings_initializer = "glorot_normal") %>% 
      layer_flatten()
    
    ########## Hadamard product between z-space and a class conditional embedding
    h <- layer_multiply(list(latent, cls))
    
    fake_gene <- ann(h)
    
    keras_model(list(latent, drug_class), fake_gene)
  }
  build_discriminator <- function(){
    ann <- keras_model_sequential()
    ann %>% layer_dense(units = 30, input_shape = ncomp, activation = "relu")
    
    input <- layer_input(shape = ncomp)
    features <- ann(input)
    
    fake <- features %>% layer_dense(1, activation = "sigmoid", name = "generation")
    
    aux <- features %>% layer_dense(class_num, activation = "softmax", name = "auxiliary")
    
    keras_model(input, list(fake, aux))
  }
  
  
  discriminator <- build_discriminator()
  discriminator %>% compile(optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1), 
                            loss = list("binary_crossentropy", "sparse_categorical_crossentropy"))

  generator <- build_generator(latent_size)
  generator %>% compile(optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1), 
                        loss = "binary_crossentropy")


  latent <- layer_input(shape = list(latent_size))
  drug_class <- layer_input(shape = list(1), dtype = "int32")

  fake <- generator(list(latent, drug_class))


  freeze_weights(discriminator)
  results <- discriminator(fake)

  combined <- keras_model(list(latent, drug_class), results)
  combined %>% compile(optimizer = optimizer_adam(lr = adam_lr, beta_1 = adam_beta_1), 
                       loss = list("binary_crossentropy", "sparse_categorical_crossentropy"))


  
  
  for(epoch in 1 : epochs){
    
    num_batches <- trunc(num_train/batch_size)
    pb <- progress_bar$new(
      total = num_batches, 
      format = sprintf("epoch %s/%s :elapsed [:bar] :percent :eta", epoch, epochs),
      clear = FALSE
    )
    
    epoch_gen_loss <- NULL
    epoch_disc_loss <- NULL
    
    possible_indexes <- 1:num_train
    
    
    for(index in 1:num_batches){
      
      pb$tick()
      
      # Generate a new batch of noise
      noise <- runif(n = batch_size*latent_size, min = -1, max = 1) %>%
        matrix(nrow = batch_size, ncol = latent_size)
      
      # Get a batch of real images
      batch <- sample(possible_indexes, size = batch_size)
      possible_indexes <- possible_indexes[!possible_indexes %in% batch]
      image_batch <- train_x[batch,,drop = FALSE]
      label_batch <- train_y[batch]
      
      # Sample some labels from p_c
      sampled_labels <- sample(0:(class_num - 1), batch_size, replace = TRUE) %>%
        matrix(ncol = 1)
      
      # Generate a batch of fake images, using the generated labels as a
      # conditioner. We reshape the sampled labels to be
      # (batch_size, 1) so that we can feed them into the embedding
      # layer as a length one sequence
      generated_images <- predict(generator, list(noise, sampled_labels))
      
      X <- abind(image_batch, generated_images, along = 1)
      y <- c(rep(1L, batch_size), rep(0L, batch_size)) %>% matrix(ncol = 1)
      aux_y <- c(label_batch, sampled_labels) %>% matrix(ncol = 1)
      
      # Check if the discriminator can figure itself out
      disc_loss <- train_on_batch(
        discriminator, x = X, 
        y = list(y, aux_y)
      )
      epoch_disc_loss <- rbind(epoch_disc_loss, unlist(disc_loss))
      
      # Make new noise. Generate 2 * batch size here such that
      # the generator optimizes over an identical number of images as the
      # discriminator
      noise <- runif(2*batch_size*latent_size, min = -1, max = 1) %>%
        matrix(nrow = 2*batch_size, ncol = latent_size)
      sampled_labels <- sample(0:(class_num - 1), size = 2*batch_size, replace = TRUE) %>%
        matrix(ncol = 1)
      
      # Want to train the generator to trick the discriminator
      # For the generator, we want all the {fake, not-fake} labels to say
      # not-fake
      trick <- rep(1, 2*batch_size) %>% matrix(ncol = 1)
      
      combined_loss <- train_on_batch(
        combined, 
        list(noise, sampled_labels),
        list(trick, sampled_labels)
      )
      
      epoch_gen_loss <- rbind(epoch_gen_loss, unlist(combined_loss))
      
    }
    
    cat(sprintf("\nTesting for epoch %02d:", epoch))
    
    # Evaluate the testing loss here
    
    # Generate a new batch of noise
    noise <- runif(num_test*latent_size, min = -1, max = 1) %>%
      matrix(nrow = num_test, ncol = latent_size)
    
    # Sample some labels from p_c and generate images from them
    sampled_labels <- sample(0:(class_num - 1), size = num_test, replace = TRUE) %>%
      matrix(ncol = 1)
    generated_images <- predict(generator, list(noise, sampled_labels))
    
    X <- abind(test_x, generated_images, along = 1)
    y <- c(rep(1, num_test), rep(0, num_test)) %>% matrix(ncol = 1)
    aux_y <- c(test_y, sampled_labels) %>% matrix(ncol = 1)
    
    # See if the discriminator can figure itself out...
    discriminator_test_loss <- evaluate(
      discriminator, X, list(y, aux_y), 
      verbose = FALSE
    ) %>% unlist()
    
    discriminator_train_loss <- apply(epoch_disc_loss, 2, mean)
    
    # Make new noise
    noise <- runif(2*num_test*latent_size, min = -1, max = 1) %>%
      matrix(nrow = 2*num_test, ncol = latent_size)
    sampled_labels <- sample(0:(class_num - 1), size = 2*num_test, replace = TRUE) %>%
      matrix(ncol = 1)
    
    trick <- rep(1, 2*num_test) %>% matrix(ncol = 1)
    
    generator_test_loss = combined %>% evaluate(
      list(noise, sampled_labels),
      list(trick, sampled_labels),
      verbose = FALSE
    )
    
    generator_train_loss <- apply(epoch_gen_loss, 2, mean)
    
    
    # Generate an epoch report on performance
    row_fmt <- "\n%22s : loss %4.2f | %5.2f | %5.2f"
    cat(sprintf(
      row_fmt, 
      "generator (train)",
      generator_train_loss[1],
      generator_train_loss[2],
      generator_train_loss[3]
    ))
    cat(sprintf(
      row_fmt, 
      "generator (test)",
      generator_test_loss[1],
      generator_test_loss[2],
      generator_test_loss[3]
    ))
    
    cat(sprintf(
      row_fmt, 
      "discriminator (train)",
      discriminator_train_loss[1],
      discriminator_train_loss[2],
      discriminator_train_loss[3]
    ))
    
    cat(sprintf(
      row_fmt, 
      "discriminator (test)",
      discriminator_test_loss[1],
      discriminator_test_loss[2],
      discriminator_test_loss[3]
    ))
    
    cat("\n")
    
    test_loss <- rbind(test_loss, discriminator_test_loss)
    train_loss <- rbind(train_loss, discriminator_train_loss)
    
    
  }
  list(test_loss = test_loss, train_loss = train_loss, 
       discriminator = discriminator, generator = generator)
}
  
#################################### Main Procedure ######################################################

library(abind)
library(keras)
library(progress)



path <- "D:\\Sariel\\LAB\\FL-ML\\Data\\GAN-build\\"
oridata <- load_fl_data(path)


data_x <- (oridata[[1]][3*(1: 360)-2, ] + oridata[[1]][3*(1: 360)-1, ] + oridata[[1]][3*(1: 360), ])/3
data_y <- cat_trans(oridata[[2]][3*(1: 360)])

ext_data <- load_fl_data("D:\\Sariel\\LAB\\FL-ML\\Data\\GAN-test\\")
ext_data_x <- ext_data[[1]]
ext_data_y <- cat_trans(ext_data[[2]])

total_train_perf <- c()
total_val_perf <- c()
total_ext_perf <- c()

for(cycle_number in 49: 70){
  trial_latent_size <- (cycle_number - 1) * 2 + 50
  cur_val_perf <- c()
  cur_train_perf <- c()
  cur_ext_perf <- c()
  for(cur_cycle_number in 1: 5){
    sample <- sample(1: length(data_y), 0.8 * length(data_y), replace = F)
    datald_x <- data_dim_reduc(data_x, sample, 100)
    train_x <- datald_x[[1]]
    test_x <- datald_x[[2]]
    train_y <- data_y[sample]
    test_y <- data_y[-sample]
    ext_data_ld <- ext_data_x %*% datald_x[[3]]
    
    GAN_result <- GAN(train_x, train_y, latent_size = 50, epochs  = 200)
    cur_val_perf <- c(cur_val_perf, 
                     sum((apply(predict(GAN_result$discriminator, test_x)[[2]], 
                                1, which.max) - 1) == test_y)/length(test_y))
    cur_train_perf <- c(cur_train_perf, 
                       sum((apply(predict(GAN_result$discriminator, train_x)[[2]], 
                                  1, which.max) - 1) == train_y)/length(train_y))
    cur_ext_perf <- c(cur_ext_perf, 
                        sum((apply(predict(GAN_result$discriminator, ext_data_ld)[[2]], 
                                   1, which.max) - 1) == ext_data_y)/length(ext_data_y))
    
    cat("current_cycle_number: ", cur_cycle_number)
    }
  total_train_perf <- rbind(total_train_perf, cur_train_perf)
  total_val_perf <- rbind(total_val_perf, cur_val_perf)
  total_ext_perf <- rbind(total_ext_perf, cur_ext_perf)
  cat("total cycle number: ", cycle_number)
}  


write.csv(total_ext_perf, "D:\\Sariel\\LAB\\FL-ML\\Data\\Epoch_total_ext_perf_50-178.csv")

write.csv(total_train_perf, "D:\\Sariel\\LAB\\FL-ML\\Data\\Epoch_total_train_perf_50-178.csv")

write.csv(total_val_perf, "D:\\Sariel\\LAB\\FL-ML\\Data\\Epoch_total_val_perf_50-178.csv")
  
  ext_data <- load_fl_data("D:\\Sariel\\LAB\\FL-ML\\Data\\GAN-test\\")
  ext_data_x <- ext_data[[1]]
  ext_data_y <- cat_trans(ext_data[[2]])
  ext_data_ld <- ext_data_x %*% datald_x[[3]]
  
  
for (cycle_num in 1: 20) {
  ncomp <- 100 + (cycle_num - 1) * 5
  data_x <- oridata[[1]]
  data_y <- cat_trans(oridata[[2]])
  
  sample <- sample(1: length(data_y), 0.8 * length(data_y), replace = F)
  datald_x <- data_dim_reduc(data_x, sample, ncomp)
  train_x <- datald_x[[1]]
  test_x <- datald_x[[2]]
  train_y <- data_y[sample]
  test_y <- data_y[-sample]
  
  GAN_result <- GAN(train_x, train_y, latent_size = 5)
  val_perf_pc <- c(val_perf_pc, 
                  sum((apply(predict(GAN_result$discriminator, test_x)[[2]], 
                             1, which.max) - 1) == test_y)/length(test_y))
  train_perf_pc <- c(train_perf_pc, 
                     sum((apply(predict(GAN_result$discriminator, train_x)[[2]], 
                                1, which.max) - 1) == train_y)/length(train_y))
  cat("cycle_number: ", cycle_num)
}

  val_perf_ls <- c()
  train_perf_ls <- c()
  for (cycle_num in 1: 50) {
    latent_size <- cycle_num 
    data_x <- oridata[[1]]
    data_y <- cat_trans(oridata[[2]])
    
    sample <- sample(1: length(data_y), 0.8 * length(data_y), replace = F)
    datald_x <- data_dim_reduc(data_x, sample, ncomp = 100)
    train_x <- datald_x[[1]]
    test_x <- datald_x[[2]]
    train_y <- data_y[sample]
    test_y <- data_y[-sample]
    
    GAN_result <- GAN(train_x, train_y, latent_size, ncomp = 100, epochs = 20)
    val_perf_ls <- c(val_perf_ls, 
                     sum((apply(predict(GAN_result$discriminator, test_x)[[2]], 
                                1, which.max) - 1) == test_y)/length(test_y))
    train_perf_ls <- c(train_perf_ls, 
                       sum((apply(predict(GAN_result$discriminator, train_x)[[2]], 
                                  1, which.max) - 1) == train_y)/length(train_y))
    cat("cycle_number: ", cycle_num)
  }
  
  
  
  
  
test_path <- "D:\\Sariel\\LAB\\FL-ML\\Data\\GAN-test\\"
oritest <- load_fl_data(test_path)
dudata_x <- oritest[[1]] %*% (datald_x$load_mtrx)
dudata_y <- cat_trans(oritest[[2]])

sum((apply(predict(GAN_result$discriminator, test_x)[[2]], 1, which.max) - 1) == test_y)




 