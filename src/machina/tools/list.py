from collections.abc import Iterable



def flatten_list(nested_list: list) -> list:
    """
    Recursively flattens a nested list into a single list.

    This function takes a list that may contain other lists (at any depth)
    and returns a new list with all elements unpacked into a single, 
    one-dimensional list.

    Parameters
    ----------
    nested_list : list
        A list that may contain integers, strings, or other lists
        (possibly nested to arbitrary depth).

    Returns
    -------
    list
        A flat list containing all the elements from the nested list,
        preserving the original order.

    Examples
    --------
    >>> flatten_list([1, [2, [3, 4]], 5])
    [1, 2, 3, 4, 5]

    >>> flatten_list([[['a']], 'b', ['c', ['d']]])
    ['a', 'b', 'c', 'd']
    """

    flat = []
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat




if __name__ == "__main__":


    # flatten list
    la = [['a', 'b', 'c'], ['d', 'e', 'f']]
    lf = flatten_list(la)
    print(lf)