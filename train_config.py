data_path_config = {
    'train_data_path': '../data/sentiment_analysis_trainingset.csv',
    'valid_data_path': '../data/sentiment_analysis_validationset.csv',
    'test_data_path': '../data/sentiment_analysis_testa.csv',
    'submission_path': '../data/submission.csv',
    'embedding_path': '../word2vec/model_word_200.kv',
    'embedding_vocab_path': '../word2vec/model_word_vocab_200.kv',
    'embedding_dim': 200,
    'check_points_base_dir': '../checkpoints',
    'model_save_base_dir': '../models',
}


hierarchical_model_config = {
    'max_sentences': 240,
    'max_sentence_length': 40
}

topic_binary_model_config = {
    'max_len': 1024,
    'batch_size': 64
}


targets = [
    'location_traffic_convenience',
    'location_distance_from_business_district',
    'location_easy_to_find',
    'service_wait_time',
    'service_waiters_attitude',
    'service_parking_convenience',
    'service_serving_speed',
    'price_level',
    'price_cost_effective',
    'price_discount',
    'environment_decoration',
    'environment_noise',
    'environment_space',
    'environment_cleaness',
    'dish_portion',
    'dish_taste',
    'dish_look',
    'dish_recommendation',
    'others_overall_experience',
    'others_willing_to_consume_again']