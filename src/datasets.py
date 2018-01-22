import pandas as pd

def path2gt(file_path, dataset):

    if dataset == 'GTZAN':
        return gtzan_path2gt(file_path)

    elif dataset == 'Ballroom':
        return ballroom_path2gt(file_path)

    elif dataset == 'Extended Ballroom':
        return extended_ballroom_path2gt(file_path)

    elif dataset == 'UrbanSound8K':
        return urban_sound_path2gt(file_path)

    else:
        import ipdb; ipdb.set_trace()

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


# BALLROOM

def ballroom_path2gt(file_path):

    cut_end = file_path[:file_path.rfind('/')]
    tag = cut_end[cut_end.rfind('/')+1:]
    print(tag)

    if tag == 'ChaChaCha':
        return 0
    elif tag == 'Jive':
        return 1
    elif tag == 'Quickstep':
        return 2
    elif tag == 'Rumba':
        return 3
    elif tag == 'Samba':
        return 4
    elif tag == 'Tango':
        return 5
    elif tag == 'VienneseWaltz':
        return 6
    elif tag == 'Waltz':
        return 7
    else:
        print('Warning: did not find the corresponding ground truth (' + str(tag) + ').')
        import ipdb; ipdb.set_trace()

# EXTENDED BALLROOM

def extended_ballroom_path2gt(file_path):

    cut_end = file_path[:file_path.rfind('/')]
    tag = cut_end[cut_end.rfind('/')+1:]
    print(tag)

    if tag == 'Chacha':
        return 0
    elif tag == 'Foxtrot':
        return 1
    elif tag == 'Jive':
        return 2
    elif tag == 'Pasodoble':
        return 3
    elif tag == 'Quickstep':
        return 4
    elif tag == 'Rumba':
        return 5
    elif tag == 'Salsa':
        return 6
    elif tag == 'Samba':
        return 7
    elif tag == 'Slowwaltz':
        return 8
    elif tag == 'Tango':
        return 9
    elif tag == 'Viennesewaltz':
        return 10
    elif tag == 'Waltz':
        return 11
    elif tag == 'Wcswing':
        return 12
    else:
        print('Warning: did not find the corresponding ground truth (' + str(tag) + ').')
        import ipdb; ipdb.set_trace()

# URBAN SOUND 8K

def urban_sound_path2gt(file_path):

    tag = file_path[file_path.rfind('/')+1:]
    print(tag)
    df = pd.read_csv('/data/UrbanSound8K/metadata/UrbanSound8K.csv')
    #import ipdb; ipdb.set_trace()
    return int(df[df.slice_file_name==tag].classID)
