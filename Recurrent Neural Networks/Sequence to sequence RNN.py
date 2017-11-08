max_len=200
vocab_size=20000
batch_size=100
layer_num=3
hidden_dim=1000
nb_epoch=20
mode='train'

#X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data('europarl-v8.fi-en.en', 'europarl-v8.fi-en.fi', MAX_LEN, VOCAB_SIZE)
X = [text_to_word_sequence(x)[::-1] for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]
y = [text_to_word_sequence(y) for x, y in zip(X_data.split('\n'), y_data.split('\n')) if len(x) > 0 and len(y) > 0 and len(x) <= max_len and len(y) <= max_len]

# Finding the length of the longest sequence
X_max_len = max([len(sentence) for sentence in X])
y_max_len = max([len(sentence) for sentence in y])

# Padding zeros to make all sequences have a same length with the longest one
X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
y = pad_sequences(y, maxlen=y_max_len, dtype='int32')

dist = FreqDist(np.hstack(X))
X_vocab = dist.most_common(vocab_size-1)
dist = FreqDist(np.hstack(y))
y_vocab = dist.most_common(vocab_size-1)

# Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
X_ix_to_word = [word[0] for word in X_vocab]
X_ix_to_word.insert(0, 'ZERO')

# Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
X_ix_to_word.append('UNK')

# Create the word-to-index dictionary from the array created above
X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

# Converting each word to its index value
for i, sentence in enumerate(X):
    for j, word in enumerate(sentence):
        if word in X_word_to_ix:
            X[i][j] = X_word_to_ix[word]
        else:
            X[i][j] = X_word_to_ix['UNK']
			
X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
y = pad_sequences(y, maxlen=y_max_len, dtype='int32')

sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
for i, sentence in enumerate(word_sentences):
    for j, word in enumerate(sentence):
        sequences[i, j, word] = 1
		
model = Sequential()
model.add(Embedding(X_vocab_len, 1000, input_length=X_max_len, mask_zero=True))
model.add(LSTM(hidden_size))

model.add(RepeatVector(y_max_len))
for _ in range(num_layers):
    model.add(LSTM(hidden_size, return_sequences=True))
	
model.add(TimeDistributed(Dense(y_vocab_len)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

for k in range(k_start, NB_EPOCH+1):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    for i in range(0, len(X), 1000):
        if i + 1000 >= len(X):
            i_end = len(X)
        else:
            i_end = i + 1000
        y_sequences = process_data(y[i:i_end], y_max_len, y_word_to_ix)

        print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X)))
        model.fit(X[i:i_end], y_sequences, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2)
    model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
	
if(mode=='test')
	saved_weights = find_checkpoint_file('.')
	#model.load_weights("weights.best.hdf5")
	if(len(saved_weights==0)):
		print('system is trained')
		sys.exit()
	else:
        X_test = load_test_data('test', X_word_to_ix, MAX_LEN)
        X_test = pad_sequences(X_test, maxlen=X_max_len, dtype='int32')
        model.load_weights(saved_weights)
            
        predictions = np.argmax(model.predict(X_test), axis=2)
        sequences = []
        for prediction in predictions:
            sequence = ' '.join([y_ix_to_word(index) for index in prediction if index > 0])
            print(sequence)
            sequences.append(sequence)
        np.savetxt('test_result', sequences, fmt='%s')