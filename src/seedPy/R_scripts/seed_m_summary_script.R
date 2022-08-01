library(tidyverse)
library(readxl)

file_list = dir('../measurement_files')[grep("*.csv$", dir('../measurement_files'))]

merged_weights <- read.csv('merged_weights_edit.csv') %>% 
        select(-X)

merged_weights <- read_excel('K2112_seed_harvest_weights.xlsx', na = 'NA') %>% 
        rename_all(tolower) %>% 
        rename('subsample' = 'sub sample',
               'remainder_seed_weight_g' = 'whole_plant_seed_wt_g',
               'seed_5rac_wt_g' = '5_raceme_seed_weight_g',
               ) %>% 
        select(-c(notes, '...9')) %>% 
        mutate(total_seed_weight_g = remainder_seed_weight_g + seed_5rac_wt_g)

merged_weights$seed_5rac_wt_g <- replace_na(merged_weights$seed_5rac_wt_g, 0) 

merged_weights$image_weights_g <- rep(NA, nrow(merged_weights))

for(row in 1:nrow(merged_weights)){
        obs <- merged_weights[row,]
        if(!(is.na(obs$subsample))){
                merged_weights$image_weights[row] <- obs$subsample
        } else if(obs$seed_5rac_wt_g > 0){
                merged_weights$image_weights[row] <- obs$seed_5rac_wt_g
        } else {
                merged_weights$image_weights[row] <- obs$remainder_seed_weight_g
        }
}

seed_summary = function(data){
        summary <- data %>%
                summarise(n_seeds = n(),
                          total_area = sum(area), 
                          mean_area = mean(area),
                          seed_area_sd = sd(area),
                          mean_convex_hull_area = mean(convex_hull_area), 
                          total_perimeter = sum(perimeter), 
                          mean_perimeter = mean(perimeter), 
                          mean_seed_length = mean(length), 
                          mean_seed_width = mean(width), 
                          mean_seed_extent = mean(extent), 
                          mean_solidity = mean(solidity), 
                          mean_equi_dia = mean(equi_diameter), 
                          mean_aspect_ratio = mean(aspect_ratio))
        summary
}

results_df <- data.frame()
for(file in file_list){
        if(grepl('NULL[0-9]*', file)){
                file_name = str_extract(file, "NULL[0-9]*")
        } else {
                file_name = str_extract(file, "([0-9]*)")
        }
        data <- read.csv(paste('../measurement_files/', file, sep = ''))
        if(nrow(data)==0){
                vec <- c(rep(0, times = 13), file_name)
                results_df <- rbind(results_df, vec)
        } else {
                summary <- seed_summary(data)
                summary$accession <- file_name
                results_df <- rbind(results_df, summary)
        }
}

results_df <- results_df %>% 
        select(accession, everything()) %>% 
        mutate_at(2:14, as.numeric) %>% 
        full_join(merged_weights, by = c('accession' = 'midcoenvelope_id')) %>% 
        mutate(seeds_per_g = n_seeds/image_weights_g)

write.csv(results_df, 'K2112CO_image_analysis_results.csv')
