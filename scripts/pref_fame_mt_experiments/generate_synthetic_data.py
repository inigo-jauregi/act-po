import argparse

from src.ctrlpost.synthetic import CpoPreferenceDataGenerator
from src.ctrlpost.utils import get_csv_artifacts_from_experiment, download_csv_artifacts

def main(args):

    # conditions = {
    #     'temperature': ('>=', 2.0),
    # }
    # conditions = {
    #     'model_name': ('=', 'anthropic.claude-3-haiku-20240307-v1:0')
    # }
    conditions = None
  #synthetic_pref_fame_mt_sonnet_4.5_ic_da-es
    model_list = ['sonnet_4.5_ic']
    lng_pair_list = [(args.src, args.tgt)]
    fix_ref = [args.fixed]

    for src, tgt in lng_pair_list:

        if src == tgt:
            continue

        for model in model_list:
            for fixed in fix_ref:

                print(f'{src}-{tgt} | {fix_ref}')
                list_artifacts = get_csv_artifacts_from_experiment(args.experiment_name,
                                                                   dict_conditions=conditions)
                dict_paths = download_csv_artifacts(list_artifacts)
                list_paths = [val for _, val in dict_paths.items()]

                # data_generator = HeuristicWeakerDelta(
                #     train_data_src='./data/CoCoA_MT/train/en-de/formality-control.train90.all.en-de.en',
                #     train_data_att1='./data/CoCoA_MT/train/en-de/formality-control.train90.all.en-de.formal.de',
                #     train_data_att2='./data/CoCoA_MT/train/en-de/formality-control.train90.all.en-de.informal.de',
                #     src_lang='en',
                #     tgt_lang='de',
                #     attr1_name='formal',
                #     attr2_name='informal',
                #     save_path='./data/CoCoA_MT/train/en-de/synthetic/heuristic_weaker_delta',
                #     tokenizer='facebook/nllb-200-distilled-600M'
                # )

                ref_pref = True if fixed == 'pref' else False
                ref_rej = True if fixed == 'rej' else False

                data_generator = CpoPreferenceDataGenerator(
                    list_sample_data_paths=list_paths,
                    # list_sample_data_paths=[
                    #     # './data/CoCoA_MT/train/en-ja/synthetic/sampled_train_data/nllb_600M/zs_nllb_deterministic.csv',
                    #     # './data/CoCoA_MT/train/en-ja/synthetic/sampled_train_data/nllb_600M/pretrained_nllb_deterministic.csv',
                    #     # './data/CoCoA_MT/train/en-ja/synthetic/sampled_train_data/nllb_600M/pretrained_nllb_sample_temp1.0.csv',
                    #     # './data/CoCoA_MT/train/en-ja/synthetic/sampled_train_data/nllb_600M/pretrained_nllb_sample_temp2.0.csv',
                    #     # './data/CoCoA_MT/train/en-ja/synthetic/sampled_train_data/nllb_600M/zs_nllb_deterministic.csv',
                    #     # './data/CoCoA_MT/train/en-ja/synthetic/sampled_train_data/nllb_600M/zs_nllb_sample_temp1.0.csv',
                    #     './data/CoCoA_MT/train/en-de/synthetic/sampled_train_data/claude_3_sonnet/zs_claude_3_sonnet_temp_1.0.csv'
                    #
                    # ],
                    src_lang=src,
                    tgt_lang=tgt,
                    save_path=f'./data/PREF_FAME_MT/{src}-{tgt}/synthetic/preference_train_data/{model}_sampling_temp_ref_{fixed}',
                    criteria='translation_and_formality',
                    ref_free_eval_model=True,
                    ref_preferred=ref_pref,
                    ref_rejected=ref_rej,
                    bleu_threshold=False,
                )

                # Generate synthetic samples
                # num_samples_generate = len(data_generator.gold_triplets)
                # data_generator.generate(num_samples_generate=num_samples_generate)
                data_generator.generate()

                # Save the synthetic samples
                data_generator.save_synthetic_data()

                del data_generator

parser = argparse.ArgumentParser(description='Train a translation model')
parser.add_argument('--experiment-name', type=str, required=True,
                    help='Path to the training data')
parser.add_argument('--src', type=str, required=True,
                    help='Path to the training data')
parser.add_argument('--tgt', type=str, required=True,
                    help='Path to the training data')
parser.add_argument('--fixed', type=str, required=True,
                    help='Path to the validation data')
if __name__ == '__main__':
    my_args = parser.parse_args()
    main(my_args)
