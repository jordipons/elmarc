# GTZAN

def gtzan_path2gt(file_path):

    tag = file_path[file_path.rfind('/')+1:file_path.rfind('.', 0, -4)]
    print(tag)

    if tag == 'blues':
        return 0
    elif tag == 'classical':
        return 1
    elif tag == 'country':
        return 2
    elif tag == 'disco':
        return 3
    elif tag == 'hiphop':
        return 4
    elif tag == 'jazz':
        return 5
    elif tag == 'metal':
        return 6
    elif tag == 'pop':
        return 7
    elif tag == 'reggae':
        return 8
    elif tag == 'rock':
        return 9
    else:
        print('Warning: did not find the corresponding ground truth (' + str(tag) + ').')
        import ipdb; ipdb.set_trace()
