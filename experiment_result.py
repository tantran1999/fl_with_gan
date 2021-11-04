import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def save_images_result(exp_path, csv_path):
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    accuracies, class_recall, poison_rate = read_csv(csv_path=csv_path, attacker_label=None, poison_label=4)
    draw_results(accuracies, class_recall, poison_rate, exp_path)

def read_csv(csv_path, attacker_label, poison_label):

    accuracies = list()
    class_recall = list()
    poison_rate = list()

    df = pd.read_csv(csv_path, header=None)
    for idx, row in df.iterrows():
        accuracy = 0.0
        for i in range(10):
            if i != poison_label:
                accuracy += row[i + 12]
        accuracies.append(accuracy/9)
        class_recall.append(row[poison_label + 12])
        poison_rate.append(row[22]/1000)

    return accuracies, class_recall, poison_rate

def draw_results(accuracies, class_recall, poison_rate, exp_path):
    round = [i for i in range(200)]

    # Main task and Poison task Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(round, accuracies, label="Main task")
    plt.plot(round, class_recall, label="Poison task")
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig(exp_path + '/class_recall.png')

    # Main task accuracy and Poison success rate
    plt.figure(figsize=(10, 5))
    plt.plot(round, accuracies, label="Main task")
    plt.plot(round, poison_rate, label="Poison task")
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig(exp_path + '/poison_success_rate.png')

if __name__ == "__main__":
    save_images_result('experiment/multi_round_attack/exp_1', '3000_results.csv')