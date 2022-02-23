import argparse

def get_args():
  parser = argparse.ArgumentParser(description="Normalizing Flow")
  parser.add_argument('--config', type=str, default='master', help='Type of configuration to load')
  args = parser.parse_args()
  return args