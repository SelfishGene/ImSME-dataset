import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import OPERATIONS


def generate_equations(first_argument_num_digits, second_argument_num_digits, result_num_digits):
    print(f'Generating equations with:')
    print(f'First argument: up to {first_argument_num_digits} digits')
    print(f'Second argument: up to {second_argument_num_digits} digits')
    print(f'Result: up to {result_num_digits} digits')
    print(f'Operations: {OPERATIONS}')

    first_argument_numbers = range(0, 10 ** first_argument_num_digits)
    second_argument_numbers = range(0, 10 ** second_argument_num_digits)
    max_result = 10 ** result_num_digits - 1

    data = []
    for arg1 in tqdm(first_argument_numbers, desc="Generating equations"):
        for arg2 in second_argument_numbers:
            for op in OPERATIONS:
                if op == '+':
                    result = arg1 + arg2
                elif op == '-':
                    result = arg1 - arg2
                elif op == '*':
                    result = arg1 * arg2
                elif op == '/':
                    if arg2 != 0 and arg1 % arg2 == 0:
                        result = arg1 // arg2
                    else:
                        continue
                
                if 0 <= result <= max_result:
                    full_string = f"{arg1} {op} {arg2} = {result}"
                    data.append([arg1, op, arg2, result, full_string])

    columns = ["first argument", "operation", "second argument", "result", "full string"]
    df = pd.DataFrame(data, columns=columns)
    
    print(f'Generated a total of {df.shape[0]} equations')
    return df


def save_dataframe(df, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    df.to_csv(file_path, index=False)
    print(f'Saved dataframe to: {file_path}')


def print_basic_stats(df):
    print("\nBasic Statistics:")
    print("-----------------")
    print("Operation distribution:")
    print(df['operation'].value_counts())
    print("\nNumber of unique values:")
    for column in ['first argument', 'second argument', 'result']:
        print(f"{column}: {df[column].nunique()}")
    print("\nMost common values:")
    for column in ['first argument', 'second argument', 'result']:
        print(f"\n{column}:")
        print(df[column].value_counts().head())


if __name__ == "__main__":
    output_folder = "data/simple_math_equations_dataset"
    
    # Generate different equation sets
    equation_sets = [
        (3, 3, 3),  # 3-digit op 3-digit = 3-digit
        (4, 3, 3),  # 4-digit op 3-digit = 3-digit
        (4, 4, 2),  # 4-digit op 4-digit = 2-digit
    ]
    
    for first_digits, second_digits, result_digits in equation_sets:
        print('-' * 80)
        print(f"\nGenerating equations: {first_digits}d op {second_digits}d = {result_digits}d")
        print('-' * 15)
        df = generate_equations(first_digits, second_digits, result_digits)
        
        filename = f"simple_math_equations__{first_digits}d_op_{second_digits}d_eq_{result_digits}d__{df.shape[0]}_rows.csv"
        save_dataframe(df, output_folder, filename)
        print('-' * 80)
        print_basic_stats(df)
        print('-' * 80)

    print("\nEquation dataframe creation completed.")
