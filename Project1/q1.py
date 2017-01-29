import pandas as pd
import matplotlib.pyplot as plt

# q1
df = pd.read_csv('network_backup_dataset.csv', ',')

df1 = df[df['Week #'] <= 3]  # should select first 20 days, but now is 21 days
df1 = df1.replace({'Day of Week': {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4,
                                   'Saturday': 5, 'Sunday': 6}})
df1['Time'] = df1[['Week #', 'Day of Week', 'Backup Start Time - Hour of Day']] \
    .apply(lambda x: (x['Week #'] - 1) * 7 * 24 + x['Day of Week'] * 24 + x['Backup Start Time - Hour of Day'], axis=1)
df1 = df1[df1['Time'] <= 480]  # the first 20 days

df1 = df1.groupby('Work-Flow-ID')
for workflow_ID, group in df1:
    workflows = group.groupby('File Name')
    for file_name, workflow in workflows:
        plt.plot(workflow['Time'], workflow['Size of Backup (GB)'], label=file_name)
    plt.xlabel('Time')
    plt.ylabel('Size')
    plt.legend(loc='best')
    plt.title(workflow_ID)
    plt.savefig(workflow_ID + '.png')
    plt.show()
