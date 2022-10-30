total=int(input('Enter number of bananas at starting'))
distance=int(input('Enter distance you want to cover'))
load_capacity=int(input('Enter max load capacity of your camel'))
loss=0
start=total
#rate of eating banas =1/mile
#taking each intermediated drop point at a stride of 1
for i in range(distance):
   while start>0:
       start=start-load_capacity
#Here if condition is checking that camel doesn't move back if there is only one banana left.
       if start==1:
           loss=loss-1 #Loss is decreased because if camel tries to get remaining one banana he will lose one extra banana for covering that two miles.
#we are increasing loss because for moving backwards and forward by one mile two bananas will be lost
       loss=loss+2
#last trip camel will not go back.
   loss=loss-1
   start=total-loss
   if start==0: #Condition to check whether it is possible to take a single banana or not.
       break
print(start)
