import numpy as np

# 초기 가중치와 바이어스 설정
weights = np.array([[0.1, 0.3]])
bias = -0.5

# 학습률
learning_rate = 0.2

# 입력값과 목표값
inputs = np.array([
    [0, 0],  # In-1
    [1, 0],  # In-2
    [0, 1],  # In-3
    [1, 1]   # In-4
])

# 목표값 (AND 연산 결과)
target_outputs = np.array([0, 0, 0, 1])

# 활성화 함수 (단위 계단 함수)
def activation_function(y):
    return np.where(y >= 0, 1, 0)

# 오차 계산
def calculate_error(target, output):
    return target - output

# 가중치 업데이트
def update_weights(weights, learning_rate, error, inputs):
    return weights + learning_rate * error * inputs

# 각 epoch에 대한 계산
for epoch in range(1, 4):  # 3 epochs
    print(f"Epoch {epoch}")
    for i, (input, target) in enumerate(zip(inputs, target_outputs), 1):
        # 합성함수 값 계산
        y = np.dot(input, weights.T) + bias
        
        # 활성화 함수 적용
        z = activation_function(y)
        
        # 오차 계산
        error = calculate_error(target, z)
        
        # 가중치 조정
        weights = update_weights(weights, learning_rate, error, input)
        
        # 결과 출력
        print(f"In-{i}: y={y[0]}, z={z[0]}, error={error}, weights={weights}")
    
    print("\n")  # Epoch 사이에 공백 추가

# 최종 가중치 출력
print(f"Final weights after 3 epochs: {weights}")
