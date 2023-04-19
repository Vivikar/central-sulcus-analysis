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

via11_qc = {'fs_qc_passed':['sub-via003','sub-via004','sub-via010','sub-via013','sub-via017','sub-via019','sub-via020','sub-via023','sub-via026','sub-via028','sub-via030','sub-via033','sub-via037','sub-via043','sub-via044','sub-via046','sub-via049','sub-via050','sub-via051','sub-via052','sub-via054','sub-via059','sub-via064','sub-via065','sub-via068','sub-via069','sub-via070','sub-via072','sub-via083','sub-via085','sub-via087','sub-via088','sub-via092','sub-via097','sub-via100','sub-via101','sub-via102','sub-via108','sub-via109','sub-via114','sub-via118','sub-via119','sub-via121','sub-via122','sub-via124','sub-via125','sub-via126','sub-via127','sub-via130','sub-via131','sub-via134','sub-via139','sub-via141','sub-via142','sub-via146','sub-via149','sub-via150','sub-via151','sub-via152','sub-via153','sub-via154','sub-via156','sub-via163','sub-via165','sub-via169','sub-via171','sub-via172','sub-via173','sub-via183','sub-via185','sub-via186','sub-via187','sub-via190','sub-via193','sub-via196','sub-via197','sub-via201','sub-via205','sub-via206','sub-via208','sub-via209','sub-via212','sub-via213','sub-via214','sub-via215','sub-via217','sub-via218','sub-via219','sub-via231','sub-via234','sub-via235','sub-via238','sub-via240','sub-via248','sub-via249','sub-via250','sub-via253','sub-via254','sub-via256','sub-via269','sub-via273','sub-via274','sub-via276','sub-via277','sub-via278','sub-via279','sub-via287','sub-via288','sub-via289','sub-via293','sub-via294','sub-via297','sub-via298','sub-via299','sub-via310','sub-via311','sub-via312','sub-via313','sub-via317','sub-via319','sub-via320','sub-via323','sub-via325','sub-via326','sub-via328','sub-via337','sub-via338','sub-via339','sub-via340','sub-via343','sub-via344','sub-via348','sub-via350','sub-via351','sub-via353','sub-via354','sub-via364','sub-via365','sub-via366','sub-via368','sub-via369','sub-via372','sub-via374','sub-via375','sub-via377','sub-via378','sub-via379','sub-via380','sub-via383','sub-via384','sub-via385','sub-via387','sub-via388','sub-via389','sub-via390','sub-via391','sub-via392','sub-via393','sub-via396','sub-via399','sub-via400','sub-via401','sub-via402','sub-via403','sub-via404','sub-via407','sub-via408','sub-via410','sub-via411','sub-via426','sub-via429','sub-via431','sub-via433','sub-via435','sub-via438','sub-via439','sub-via440','sub-via441','sub-via442','sub-via443','sub-via444','sub-via450','sub-via451','sub-via452','sub-via453','sub-via456','sub-via458','sub-via459','sub-via467','sub-via471','sub-via473','sub-via476','sub-via478','sub-via481','sub-via483','sub-via486','sub-via488','sub-via489','sub-via493','sub-via494','sub-via495','sub-via499','sub-via501','sub-via504','sub-via506','sub-via509','sub-via510','sub-via511','sub-via512','sub-via513','sub-via515','sub-via517','sub-via518','sub-via519','sub-via521','sub-via522'],
            'bvisa_qc_passed': ['sub-via004','sub-via005','sub-via020','sub-via023','sub-via038','sub-via052','sub-via053','sub-via069','sub-via070','sub-via072','sub-via081','sub-via083','sub-via085','sub-via088','sub-via090','sub-via098','sub-via101','sub-via108','sub-via109','sub-via118','sub-via124','sub-via125','sub-via126','sub-via130','sub-via141','sub-via142','sub-via146','sub-via149','sub-via150','sub-via151','sub-via152','sub-via153','sub-via160','sub-via161','sub-via162','sub-via168','sub-via171','sub-via173','sub-via179','sub-via185','sub-via186','sub-via201','sub-via205','sub-via206','sub-via209','sub-via212','sub-via213','sub-via215','sub-via217','sub-via224','sub-via234','sub-via253','sub-via261','sub-via269','sub-via273','sub-via276','sub-via277','sub-via279','sub-via281','sub-via283','sub-via289','sub-via294','sub-via319','sub-via320','sub-via323','sub-via325','sub-via326','sub-via328','sub-via330','sub-via336','sub-via343','sub-via344','sub-via348','sub-via350','sub-via354','sub-via355','sub-via358','sub-via362','sub-via363','sub-via364','sub-via368','sub-via372','sub-via374','sub-via377','sub-via378','sub-via379','sub-via385','sub-via387','sub-via389','sub-via390','sub-via391','sub-via393','sub-via399','sub-via400','sub-via401','sub-via402','sub-via407','sub-via408','sub-via410','sub-via412','sub-via416','sub-via429','sub-via435','sub-via438','sub-via442','sub-via443','sub-via444','sub-via452','sub-via453','sub-via456','sub-via467','sub-via478','sub-via481','sub-via483','sub-via486','sub-via488','sub-via489','sub-via492','sub-via499','sub-via504','sub-via509','sub-via513','sub-via515','sub-via517','sub-via522']
}