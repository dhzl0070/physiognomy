import pandas as pd

home_path = r'D:\workplace\test\shape\ex\train2'

arch_df = pd.read_pickle(home_path + '\\arch.pkl')
deep_arch_df = pd.read_pickle(home_path + '\\deep_arch.pkl')
flat_df = pd.read_pickle(home_path + '\\flat.pkl')
up_df = pd.read_pickle(home_path + '\\up.pkl')

colum_names = ['ef0', 'ef1', 'ef2', 'ef3', 'ef4', 'rad0', 'rad1', 'rad2', 'rad3',
               'rad_ratio0', 'rad_ratio1', 'rad_ratio2']

len(arch_df) # 547
len(deep_arch_df) # 547
len(flat_df) # 547
len(up_df) # 547

deep_arch_df['eyebrow_shape'] = float(0)
arch_df = pd.concat([arch_df, deep_arch_df])
arch_df.reset_index(inplace=True)
flat_df['eyebrow_shape'] = float(1)
up_df['eyebrow_shape'] = float(2)

# arch_df = arch_df.loc[:449]
# deep_arch_df = deep_arch_df.loc[:449]
# flat_df = flat_df.loc[:449]
# up_df = up_df.loc[:449]

arch_df['ef2'].describe()
flat_df['ef2'].describe()
up_df['ef2'].describe()

arch_df['ef3'].describe()
flat_df['ef3'].describe()
up_df['ef3'].describe()

arch_df['rad0'].describe()
arch_df.drop(arch_df[arch_df['rad0'] > 0.22].index, inplace=True)
flat_df['rad0'].describe()
flat_df.drop(flat_df[flat_df['rad0'] > 0.14].index, inplace=True)
up_df['rad0'].describe()
up_df.drop(up_df[up_df['rad0'] < 0.19].index, inplace=True)

arch_df['rad1'].describe()
flat_df['rad1'].describe()
flat_df.drop(flat_df[flat_df['rad1'] > 0.03].index, inplace=True)
up_df['rad1'].describe()
up_df.drop(up_df[up_df['rad1'] < 0.19].index, inplace=True)

arch_df['rad3'].describe()
arch_df.drop(arch_df[arch_df['rad3'] < -1.60].index, inplace=True)
flat_df['rad3'].describe()
flat_df.drop(flat_df[flat_df['rad3'] < -1.21].index, inplace=True)
up_df['rad3'].describe()
up_df.drop(up_df[up_df['rad3'] > -0.60].index, inplace=True)

arch_df['rad_ratio0'].describe()
arch_df.drop(arch_df[arch_df['rad_ratio0'] < 35].index, inplace=True)
flat_df['rad_ratio0'].describe()
flat_df.drop(flat_df[flat_df['rad_ratio0'] > 150].index, inplace=True)
up_df['rad_ratio0'].describe()
up_df.drop(up_df[up_df['rad_ratio0'] > 24].index, inplace=True)

arch_df.reset_index(inplace=True)
flat_df.reset_index(inplace=True)
up_df.reset_index(inplace=True)

arch_df.to_pickle(home_path + '\\arch_.pkl')
flat_df.to_pickle(home_path + '\\flat_.pkl')
up_df.to_pickle(home_path + '\\up_.pkl')
