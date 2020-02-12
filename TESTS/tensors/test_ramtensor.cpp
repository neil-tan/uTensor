#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "RamTensor.hpp"

#include <iostream>
using std::cout;
using std::endl;

using namespace uTensor;

void setup_context(){
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
}

TEST(RAM_Tensor, constructor) {
  //setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
}

TEST(RAM_Tensor, read_write_u8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2,2) = (uint8_t) 5;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, read_write_u8_multi_tensor) {
  ///setup_context();
  localCircularArenaAllocator<512> meta_allocator;
  localCircularArenaAllocator<512> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r1({10, 10}, u8);
  RamTensor r2({10, 10}, u8);
  RamTensor r3({10, 10}, u8);
  r1(2,2) = (uint8_t) 5;
  r2(2,2) = (uint8_t) 5;
  r3(2,2) = (uint8_t) r1(2,2) + (uint8_t) r2(2,2);
  EXPECT_EQ((uint8_t)r3(2,2), 10);
  cout << "Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, read_write_u8_2x) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2,2) = (uint8_t) 5;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  r(2,2) = (uint8_t) 15;
  EXPECT_EQ((uint8_t)r(2,2), 15);
}

TEST(RAM_Tensor, read_write_u8_contig) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u8);
  r(2,2) = (uint8_t) 5;
  r(3,2) = (uint8_t) 35;
  uint8_t read = r(2,2);
  EXPECT_EQ(read, 5);
  r(2,2) = (uint8_t) 15;
  EXPECT_EQ((uint8_t)r(2,2), 15);
  EXPECT_EQ((uint8_t)r(3,2), 35);
}

TEST(RAM_Tensor, read_write_i8) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, i8);
  r(2,2) = (int8_t) -5;
  int8_t read = r(2,2);
  EXPECT_EQ(read, -5);
  cout << "i8 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
  cout << "Sizeof RamTensor " << sizeof(r) << endl;
}

TEST(RAM_Tensor, read_write_u16) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, u16);
  r(2,2) = (uint16_t) 5;
  r(3,2) = (uint16_t) 15;
  uint16_t read = r(2,2);
  EXPECT_EQ(read, 5);
  read = r(3,2);
  EXPECT_EQ(read, 15);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}

TEST(RAM_Tensor, read_write_i16) {
  ///setup_context();
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  RamTensor r({10, 10}, i16);
  r(2,2) = (int16_t) 5;
  r(3,2) = (int16_t) -15;
  int16_t read = r(2,2);
  EXPECT_EQ(read, 5);
  read = r(3,2);
  EXPECT_EQ(read, -15);
  cout << "uint16 Sizeof IntegralValue " << sizeof(IntegralValue(5)) << endl;
}
