# MANUAL SPLIT OF BRAIN VISA DATASET INTO TRANING, VALIDATION AND TEST SETS

bvisa_splits = {'train':['s12532',
                         's12636',
                         'sujet06',
                         'hades',
                         's12898',
                         'poseidon',
                         'icbm201T',
                         'icbm310T',
                         'icbm100T',
                         'icbm125T',
                         's12562',
                         'cronos',
                         's12431',
                         'icbm300T',
                         's12401',
                         'icbm200T',
                         'jupiter',
                         'isis',
                         'sujet09',
                         's12300',
                         's12913',
                         'demeter',
                         'vishnu',
                         's12635',
                         's12920',
                         'jah2',
                         'sujet10',
                         'neptune',
                         'athena',
                         'sujet12',
                         'osiris',
                         'caca',
                         'anubis',
                         's12590',
                         'horus',
                         's12919',
                         's12539',
                         'sujet02'],
                'validation':['eros',
                              'icbm320T',
                              'moon',
                              'ra',
                              's12158',
                              's12258',
                              's12277',
                              'shiva',
                              'sujet04',
                              'sujet05',
                              'sujet11',
                              'zeus'],
                'test':['ammon',
                        'atlas',
                        'beflo',
                        'hyperion',
                        'jason',
                        's12508',
                        's12826',
                        'sujet01',
                        'sujet03',
                        'sujet07',
                        'sujet08',
                        'vayu']}
###############################
# BRAIN VISA SULCI LABELS AND NAMES PER ###########################
# it total 64 sulci labels per hemisphere
# -100 in bvisa_right_sulci_idx means that it is not present in the dataset
bvisa_left_sulci_labels = [36, 32, 49, 64, 126, 3, 55, 46, 22, 28, 7, 21, 39, 41, 44, 9,
                           60, 6, 12, 48, 125, 134, 42, 29, 20, 26, 10, 54, 40, 34, 53, 4,
                           45, 47, 25, 62, 51, 24, 59, 13, 57, 8, 11, 18, 58, 129, 23, 38,
                           5, 37, 52, 16, 27, 30, 61, 50, 1, 35, 31, 43, 19, 130, 33, 17]

bvisa_right_sulci_labels = [104, 71, 118, 124, 132, 122, 133, 120, 97, 114, 68, 79, 72, 91,
                            127, 95, 89, 84, 92, 70, 115, 135, 86, 123, 103, 101, 65, 110,
                            90, 85, 107, 88, 69, 100, 108, 98, 111, 100, 112, 106, 67, 78,
                            77, 96, 128, 93, 99, 66, 73, 117, 113, 94, 116, 83, 82, 80,
                            109, 74, 76, 105, 121, 131, 81, 87]
bvisa_sulci_names = ['F.C.L.a.', 'F.C.L.p.', 'F.C.L.r.ant.', 'F.C.L.r.asc',
                     'F.C.L.r.diag.', 'F.C.L.r.retroC.tr', 'F.C.L.r.sc.ant.',
                     'F.C.L.r.sc.post.', 'F.C.M.ant.', 'F.C.M.post.',
                     'F.Cal.ant.-Sc.Cal.', 'F.Coll.', 'F.I.P.', 'F.I.P.Po.C.inf',
                     'F.I.P.r.int.1', 'F.I.P.r.int.2', 'F.P.O.', 'INSULA', 'OCCIPITAL',
                     'S.C.', 'S.C.LPC.', 'S.C.sylvian.', 'S.Call.', 'S.Cu.', 'S.F.inf.',
                     'S.F.inf.ant.', 'S.F.int.', 'S.F.inter.', 'S.F.marginal.',
                     'S.F.median.', 'S.F.orbitaire.', 'S.F.polaire.tr', 'S.F.sup.',
                     'S.GSM.', 'S.Li.ant.', 'S.Li.post.', 'S.O.T.lat.ant.',
                     'S.O.T.lat.int.', 'S.O.T.lat.med.', 'S.O.T.lat.post.', 'S.O.p.',
                     'S.Olf.', 'S.Or.', 'S.Pa.int.', 'S.Pa.sup.', 'S.Pa.t.',
                     'S.Pe.C.inf.', 'S.Pe.C.inter.', 'S.Pe.C.marginal.',
                     'S.Pe.C.median.', 'S.Pe.C.sup.', 'S.Po.C.sup.', 'S.R.inf.',
                     'S.R.sup.', 'S.Rh.', 'S.T.i.ant.', 'S.T.i.post.', 'S.T.pol.',
                     'S.T.s.', 'S.T.s.ter.asc.ant.', 'S.T.s.ter.asc.post.',
                     'S.intraCing.', 'S.p.C.', 'S.s.P.']

bvisa_padding_dims = {'skull_stripped': {'original': (160, 224, 192),
                                         '[2, 2, 2]': (96, 96, 96)},

                      'left_skeleton': {'original': (160, 224, 96),
                                        '[2, 2, 2]': (96, 96, 64)},

                      'right_skeleton': {'original': (160, 224, 96),
                                         '[2, 2, 2]': (96, 96, 64)},

                      'sulci_skeletons': {'original': (160, 224, 192),
                                          '[2, 2, 2]': (96, 96, 96)}}
#######################

synthseg_sst_splits = {'val': ['training_seg_02',
                               'training_seg_05',
                               'training_seg_11',
                               'training_seg_17']}

bad_via11 = ['sub-via419', # missing folds data [CFIN]
             'sub-via510', # missing folds data [CFIN]
             'sub-via244', # missing folds data [DRCMR]
             'sub-via286', # missing folds data [DRCMR]
             ]