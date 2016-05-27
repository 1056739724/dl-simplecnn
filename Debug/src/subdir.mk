################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../src/weight_init.cc 

CPP_SRCS += \
../src/main.cpp \
../src/matoperator.cpp \
../src/otheroperator.cpp \
../src/readData.cpp \
../src/supportOperator.cpp \
../src/trainNet.cpp 

CC_DEPS += \
./src/weight_init.d 

OBJS += \
./src/main.o \
./src/matoperator.o \
./src/otheroperator.o \
./src/readData.o \
./src/supportOperator.o \
./src/trainNet.o \
./src/weight_init.o 

CPP_DEPS += \
./src/main.d \
./src/matoperator.d \
./src/otheroperator.d \
./src/readData.d \
./src/supportOperator.d \
./src/trainNet.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


