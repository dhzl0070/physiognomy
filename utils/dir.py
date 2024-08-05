
save_home = r'D:\workplace\test\shape\valid'
none = save_home + r'\none'

folder_name = ['faceshape', 'eyeshape', 'eyetailshape', 'eyedistance',
               'eyebrowshape', 'noseshape', 'mouthshape', 'mouthtailshape']

face_shape_classlst = ['heart', 'oval', 'round', 'square', 'dia', 'long']
eye_shape_classlst = ["round", "big", "normal", "small", "thin"]
eyetail_shape_classlst = ["down", "normal", "up"]
eye_distance_classlst = ["distant", "normal", "close"]
eyebrow_shape_classlst = ['arch', 'flat', 'up']
nose_shape_classlst = ['big', 'small', 'long', 'short', 'normal']
mouth_shape_classlst = ['big','small','upper thick', 'lower thick', 'thick', 'thin', 'normal']
mouthtail_shape_classlst = ['up', 'normal', 'down']

face_shape_heart = f'{save_home}\{folder_name[0]}\{face_shape_classlst[0]}'
face_shape_oval = f'{save_home}\{folder_name[0]}\{face_shape_classlst[1]}'
face_shape_round = f'{save_home}\{folder_name[0]}\{face_shape_classlst[2]}'
face_shape_square = f'{save_home}\{folder_name[0]}\{face_shape_classlst[3]}'
face_shape_dia = f'{save_home}\{folder_name[0]}\{face_shape_classlst[4]}'
face_shape_long = f'{save_home}\{folder_name[0]}\{face_shape_classlst[5]}'
face_shape_none = f'{save_home}\{folder_name[0]}' + r'\none'

eye_shape_round = f'{save_home}\{folder_name[1]}\{eye_shape_classlst[0]}'
eye_shape_big = f'{save_home}\{folder_name[1]}\{eye_shape_classlst[1]}'
eye_shape_normal = f'{save_home}\{folder_name[1]}\{eye_shape_classlst[2]}'
eye_shape_small = f'{save_home}\{folder_name[1]}\{eye_shape_classlst[3]}'
eye_shape_thin = f'{save_home}\{folder_name[1]}\{eye_shape_classlst[4]}'
eye_shape_none = f'{save_home}\{folder_name[1]}' + r'\none'

eye_tail_down = f'{save_home}\{folder_name[2]}\{eyetail_shape_classlst[0]}'
eye_tail_normal = f'{save_home}\{folder_name[2]}\{eyetail_shape_classlst[1]}'
eye_tail_up = f'{save_home}\{folder_name[2]}\{eyetail_shape_classlst[2]}'
eye_tail_none = f'{save_home}\{folder_name[2]}' + r'\none'

eye_distance_distant = f'{save_home}\{folder_name[3]}\{eye_distance_classlst[0]}'
eye_distance_normal = f'{save_home}\{folder_name[3]}\{eye_distance_classlst[1]}'
eye_distance_close = f'{save_home}\{folder_name[3]}\{eye_distance_classlst[2]}'
eye_distance_none = f'{save_home}\{folder_name[3]}' + r'\none'

eyebrow_shape_arch = f'{save_home}\{folder_name[4]}\{eyebrow_shape_classlst[0]}'
eyebrow_shape_flat = f'{save_home}\{folder_name[4]}\{eyebrow_shape_classlst[1]}'
eyebrow_shape_up = f'{save_home}\{folder_name[4]}\{eyebrow_shape_classlst[2]}'
eyebrow_shape_none = f'{save_home}\{folder_name[4]}' + r'\none'

nose_shape_big = f'{save_home}\{folder_name[5]}\{nose_shape_classlst[0]}'
nose_shape_small = f'{save_home}\{folder_name[5]}\{nose_shape_classlst[1]}'
nose_shape_long = f'{save_home}\{folder_name[5]}\{nose_shape_classlst[2]}'
nose_shape_short = f'{save_home}\{folder_name[5]}\{nose_shape_classlst[3]}'
nose_shape_normal = f'{save_home}\{folder_name[5]}\{nose_shape_classlst[4]}'
nose_shape_none = f'{save_home}\{folder_name[5]}' + r'\none'

mouth_shape_big = f'{save_home}\{folder_name[6]}\{mouth_shape_classlst[0]}'
mouth_shape_small = f'{save_home}\{folder_name[6]}\{mouth_shape_classlst[1]}'
mouth_shape_upperthick = f'{save_home}\{folder_name[6]}\{mouth_shape_classlst[2]}'
mouth_shape_lowerthick = f'{save_home}\{folder_name[6]}\{mouth_shape_classlst[3]}'
mouth_shape_thick = f'{save_home}\{folder_name[6]}\{mouth_shape_classlst[4]}'
mouth_shape_thin = f'{save_home}\{folder_name[6]}\{mouth_shape_classlst[5]}'
mouth_shape_normal = f'{save_home}\{folder_name[6]}\{mouth_shape_classlst[5]}'
mouth_shape_none = f'{save_home}\{folder_name[6]}' + r'\none'

mouthtail_shape_up = f'{save_home}\{folder_name[7]}\{mouthtail_shape_classlst[0]}'
mouthtail_shape_down = f'{save_home}\{folder_name[7]}\{mouthtail_shape_classlst[2]}'
mouthtail_shape_none = f'{save_home}\{folder_name[7]}' + r'\none'
