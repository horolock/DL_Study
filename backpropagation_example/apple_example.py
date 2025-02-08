from layer_naive import MulLayer

########################
# Example input
# Apple's price is 100
# Going to buy 2 apples
# Tax is 10%
########################
apple_price = 100
number_of_apple = 2
tax = 1.1

#########################
# Layers
# There will be 2 layers
# - Apple's price layer
# - Tax Layer
#########################
mul_apple_price_layer = MulLayer()
mul_tax_layer = MulLayer()

######################
# Forward propagation
######################
fp_apple_price = mul_apple_price_layer.forward(apple_price, number_of_apple)
fp_final_price = mul_tax_layer.forward(fp_apple_price, tax)

print(fp_final_price)

####################################
# Backward propagation
# bp_total_price : Before tax price
####################################
bp_price = 1
bp_total_price, bp_tax = mul_tax_layer.backward(bp_price)
bp_apple_price, bp_number_of_apple = mul_apple_price_layer.backward(bp_total_price)

print(bp_total_price, bp_tax, bp_apple_price, bp_number_of_apple)