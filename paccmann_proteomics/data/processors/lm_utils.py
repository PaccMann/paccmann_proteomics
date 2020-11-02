from itertools import islice

def split_into_chunks(items, iterable):
    """Split into chunks every `items` number of lines

    Args:
        items ([type]): number of lines per chunk
        iterable ([type]): object to be chunked

    Yields:
        [iterator]: generator
    
    Example:
    >>> list(split_every(5, range(9)))
    [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    """
    i = iter(iterable)
    piece = list(islice(i, items))
    while piece:
        yield piece
        piece = list(islice(i, items))
