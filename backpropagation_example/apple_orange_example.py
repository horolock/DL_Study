from layer_naive import MulLayer, AddLayer

# Input
apple_price = 100
orange_price = 150
apple_count = 2
orange_count = 3
tax = 1.1

# Layers
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# Forward propagation
total_apple_price_before_tax = mul_apple_layer.forward(apple_price, apple_count)
total_orange_price_before_tax = mul_orange_layer.forward(orange_price, orange_count)
total_price_before_tax = add_apple_orange_layer.forward(total_apple_price_before_tax, total_orange_price_before_tax)
total_price_after_tax = mul_tax_layer.forward(total_price_before_tax, tax)

print(total_price_after_tax)

# Backward propagation
dprice = 1
print("### [BP] Before tax price * Tax ###")
d_before_tax_price, d_tax = mul_tax_layer.backward(dprice)
print(d_before_tax_price, d_tax)

print("### [BP] Apple total + Orange total ###")
d_total_apple_price, d_total_orange_price = add_apple_orange_layer.backward(d_before_tax_price)
print(d_total_apple_price, d_total_orange_price)

print("### [BP] Apple price * Apple count ###")
d_apple_price, d_apple_count = mul_apple_layer.backward(d_total_apple_price)
print(d_apple_price, d_apple_count)

print("### [BP] Orange price * Orange count ###")
d_orange_price, d_orange_count = mul_orange_layer.backward(d_total_orange_price)
print(d_orange_price, d_orange_count)
