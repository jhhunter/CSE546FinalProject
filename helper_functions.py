import matplotlib.pyplot as plt

def plot_train_rewards(environment_name, algorithm_name, episodic_rewards, ymin, ymax, show=False):
    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(episodic_rewards, 'ro')
    plt.xlabel('Episode', fontsize=28)
    plt.ylabel('Reward Value', fontsize=28)
    plt.title('Reward per episode', fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.grid()
    if(show):
        plt.show()
    else:
        plt.savefig(f'images/train_rewards_{environment_name}_{algorithm_name}.png')

def plot_test_rewards(environment_name, algorithm_name, test_rewards, ymin, ymax, show=False):
    plt.clf()
    plt.figure(figsize=(15,10))
    plt.plot(test_rewards, 'ro')
    plt.xlabel('Episode', fontsize=28)
    plt.ylabel('Reward Value', fontsize=28)
    plt.title('Reward per episode', fontsize=36)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.grid()
    if(show):
        plt.show()
    else:
        plt.savefig(f'images/test_rewards_{environment_name}_{algorithm_name}.png')