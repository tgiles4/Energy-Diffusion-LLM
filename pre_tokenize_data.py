"""Materialize tokenized train/valid caches used by training (same paths as main.py).

If the cache already exists, get_dataset loads from disk and returns immediately.
Run on a CPU node to avoid GPU-idle policies during Hugging Face preprocessing.

Example:
  python pre_tokenize_data.py hydra.job.chdir=false \\
    ++path=/scratch/$USER/edlm data=openwebtext model=small model.length=1024
"""

import hydra
import omegaconf

import dataloader


def _validation_split_name(config):
  if config.data.valid in ['text8', 'lm1b', 'ag_news']:
    return 'test'
  return 'validation'


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: omegaconf.DictConfig) -> None:
  tokenizer = dataloader.get_tokenizer(config)
  nproc, mbs, wbs = dataloader.resolve_preprocess_parallelism(config)
  dataloader.get_dataset(
    config.data.train,
    tokenizer,
    mode='train',
    wrap=config.data.wrap,
    cache_dir=config.data.cache_dir,
    block_size=config.model.length,
    streaming=config.data.streaming,
    num_proc=nproc,
    map_batch_size=mbs,
    writer_batch_size=wbs,
  )
  dataloader.get_dataset(
    config.data.valid,
    tokenizer,
    wrap=config.data.wrap,
    mode=_validation_split_name(config),
    cache_dir=config.data.cache_dir,
    block_size=config.model.length,
    streaming=False,
    num_proc=nproc,
    map_batch_size=mbs,
    writer_batch_size=wbs,
  )


if __name__ == '__main__':
  main()
