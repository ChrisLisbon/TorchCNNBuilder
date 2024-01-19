import sys


def setup_notebook():
    """
    Adding the absolute path of the entire project directory
    """
    # going back two directories
    absolute_path = '/'.join(sys.path[0].split('/')[:-2])
    sys.path.append(absolute_path)


if __name__ == '__main__':
    setup_notebook()
