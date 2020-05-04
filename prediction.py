import torch
import csv
import nltk
import os

import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('stopwords')

from GAN.train import evaluate_cycle_gan
from GAN.CycleGan import get_cycle_gan_network, get_criterions, get_optimizers, get_schedulers
from Utils.DatasetLoader import load_dataset
from Utils.Services import get_sentence_from_tensor
from constants import DATA_FOLDER, POSITIVE_FILE_EXTENSION, NEGATIVE_FILE_EXTENSION, DATASET_TYPES, TEST_TYPE, VALIDATION_TYPE


batch_size = 16
stop_words = stopwords.words('english')

def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''
        <script src='/static/components/requirejs/require.js'></script>
        <script>
            requirejs.config({
                paths: {
                    base: '/static/base',
                    plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
            });
        </script>
        '''))

def get_device():
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print('device:', device.type)
    return device

def average_words(x):
    words = x.split()
    return sum(len(word) for word in words) / len(words)

def explore_the_data(DATASET_TYPES, POSITIVE_FILE_EXTENSION, NEGATIVE_FILE_EXTENSION):
    for dataset_type in DATASET_TYPES.keys():
        for extension in [POSITIVE_FILE_EXTENSION, NEGATIVE_FILE_EXTENSION]:
            dataset = pd.read_csv(f'{DATA_FOLDER}/{DATASET_TYPES[dataset_type]}{extension}', delimiter='\t', names=['sentence'])
            dataset['word_count'] = dataset['sentence'].apply(lambda x: len(x.split()))
            dataset['char_count'] = dataset['sentence'].apply(lambda x: np.sum([len(ch) for ch in x.split()]))
            dataset['average_word_length'] = dataset['sentence'].apply(average_words)
            dataset['stopword_count'] = dataset['sentence'].apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))
            dataset['stopword_rate'] = dataset['stopword_count'] / dataset['word_count']

            dataset_by_word_count = pd.DataFrame(dataset.groupby(['word_count']).count()).reset_index()

            print(f'Display sentences info and statistics for dataset type: {DATASET_TYPES[dataset_type]}{extension}\n\n')
            print(f'dataset size: {dataset.shape[0]}\n')

            print(dataset.head(10))

            print('\n\nsentences statics')
            print(dataset.describe())

            print('\n\nnumber of sentences per word count')
            print(dataset_by_word_count[['word_count', 'sentence']])

def prepare_the_networks(G_INPUT_DIM, G_OUTPUT_DIM, device, PAD_IDX, SOS_IDX):
    g_ab, g_ba, d_a, d_b = get_cycle_gan_network(G_INPUT_DIM, G_OUTPUT_DIM, device, PAD_IDX, SOS_IDX, True, True)
    return g_ab, g_ba, d_a, d_b

def display_the_models(g_ab, d_a):
    print('generator network', g_ab)
    print('discriminator network', d_a)

def test_the_generators(PAD_IDX, device, source, g_ab, g_ba, d_a, d_b, iterators):
    criterion_g_ab, criterion_g_ba, criterion_gan, criterion_descriminator, criterion_cycle, criterion_identity = get_criterions(
        PAD_IDX, device)

    loss_gan_ab, loss_gan_ba, bleu_score_a, bleu_score_b = evaluate_cycle_gan(
        source,
        device,
        g_ab,
        g_ba,
        d_a,
        d_b,
        iterators[2], #test data
        criterion_gan)
    print(f'\nloss_gan_ab: {loss_gan_ab} | loss_gan_ba: {loss_gan_ba} | bleu_score_a: {bleu_score_a} | bleu_score_b: {bleu_score_b}')

def display_results(d_a, d_b, test_iterator, device, g_ab, g_ba, source):
    res_ab = []
    res_ba = []

    d_a.eval()
    d_b.eval()
    smoother = SmoothingFunction()

    with torch.no_grad():
        with tqdm(total=len(test_iterator)) as pbar:
            for i, batch in enumerate(test_iterator):
                pbar.update(1)

                # set model input
                real_a = batch.src.to(device)
                real_b = batch.trg.to(device)

                _, fake_b = g_ab(real_a, 0)
                _, fake_a = g_ba(real_b, 0)

                # save a to b scores
                real_a_sentences = get_sentence_from_tensor(source, real_a)
                fake_b_sentences = get_sentence_from_tensor(source, fake_b)
                for real_a_sentence, fake_b_sentence in zip(real_a_sentences, fake_b_sentences):
                    bleu_score = sentence_bleu([real_a_sentence], fake_b_sentence, smoothing_function=smoother.method4) * 100
                    res_ab.append({'Original': ' '.join(real_a_sentence), 'Transformed': ' '.join(fake_b_sentence), 'Bleu_Score': bleu_score})

                # save b to a scores
                real_b_sentences = get_sentence_from_tensor(source, real_b)
                fake_a_sentences = get_sentence_from_tensor(source, fake_a)
                for real_b_sentence, fake_a_sentence in zip(real_b_sentences, fake_a_sentences):
                    bleu_score = sentence_bleu([real_b_sentence], fake_a_sentence, smoothing_function=smoother.method4) * 100
                    res_ba.append({'Original': ' '.join(real_b_sentence), 'Transformed': ' '.join(fake_a_sentence), 'Bleu_Score': bleu_score})

    return res_ab, res_ba

def generator_positive_to_negative_results(res_ab):
    df_ab = pd.DataFrame(res_ab)
    df_ab['Original_Length'] = df_ab['Original'].apply(lambda x: len(x.split()))
    df_ab_bleu_by_length = pd.DataFrame(df_ab.groupby(['Original_Length'])['Bleu_Score'].mean())
    df_ab_bleu_by_length = df_ab_bleu_by_length.reset_index()
    df_ab_bleu_by_length['network_type'] = 'G_AB'
    print(df_ab.head(20))

    print(df_ab[(df_ab['Bleu_Score'] < 40) & (df_ab['Original_Length'] < 8) & (~df_ab['Original'].str.contains('<unk>')) & (~df_ab['Transformed'].str.contains('<unk>'))].sort_values(by=['Bleu_Score'], ascending=False).head(20))

    return df_ab_bleu_by_length

def generator_negative_to_positive_results(res_ba):
    df_ba = pd.DataFrame(res_ba)
    df_ba['Original_Length'] = df_ba['Original'].apply(lambda x: len(x.split()))
    df_ba_bleu_by_length = pd.DataFrame(df_ba.groupby(['Original_Length'])['Bleu_Score'].mean())
    df_ba_bleu_by_length = df_ba_bleu_by_length.reset_index()
    df_ba_bleu_by_length['network_type'] = 'G_BA'
    print(df_ba.head(20))

    print(df_ba[(df_ba['Bleu_Score'] < 40) & (df_ba['Original_Length'] < 8) & (~df_ba['Original'].str.contains('<unk>')) & (~df_ba['Transformed'].str.contains('<unk>'))].sort_values(by=['Bleu_Score'], ascending=False).head(20))
    return df_ba_bleu_by_length

def bleu_score(df_ab_bleu_by_length, df_ba_bleu_by_length):
    df_ab_disply = df_ab_bleu_by_length[(df_ab_bleu_by_length['Original_Length'] <= 15) & (df_ab_bleu_by_length['Original_Length'] >= 2)]
    df_ba_disply = df_ba_bleu_by_length[(df_ba_bleu_by_length['Original_Length'] <= 15) & (df_ba_bleu_by_length['Original_Length'] >= 2)]

    configure_plotly_browser_state()
    ab_bat = go.Bar(
        x=df_ab_disply['Original_Length'],
        y=df_ab_disply['Bleu_Score'],
        name='G_AB')

    ba_bar = go.Bar(
        x=df_ba_disply['Original_Length'],
        y=df_ba_disply['Bleu_Score'],
        name='G_BA')

    data = [ab_bat, ba_bar]

    layout = go.Layout(
        title='Bleu score per sentence length',
        xaxis={'title': 'Sentence Length'},
        yaxis={'title': 'Bleu Score'})

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)

    print(df_ab_disply[['Original_Length', 'Bleu_Score']])

    print(df_ba_disply[['Original_Length', 'Bleu_Score']])

def main():
    device = get_device()
    source, iterators = load_dataset(batch_size, device)
    train_iterator, validation_iterator, test_iterator = iterators
    G_INPUT_DIM = len(source.vocab)
    G_OUTPUT_DIM = len(source.vocab)
    SOS_IDX = source.vocab.stoi['<sos>']
    PAD_IDX = source.vocab.stoi['<pad>']

    #explore_the_data()
    g_ab, g_ba, d_a, d_b = prepare_the_networks(G_INPUT_DIM, G_OUTPUT_DIM, device, PAD_IDX, SOS_IDX)
    #display_the_models(g_ab, d_a)
    #test_the_generators(PAD_IDX, device, source, g_ab, g_ba, d_a, d_b, iterators)
    res_ab, res_ba = display_results(d_a, d_b, test_iterator, device, g_ab, g_ba, source)
    df_ab_bleu_by_length = generator_positive_to_negative_results(res_ab)
    df_ba_bleu_by_length = generator_negative_to_positive_results(res_ba)
    bleu_score(df_ab_bleu_by_length, df_ba_bleu_by_length)

if __name__ == '__main__':
    main()