#ifndef __UTENSOR_TYPES_H
#define __UTENSOR_TYPES_H
#include <array>
#include <cstdint>

using std::array;

class TensorShape {
 public:
  TensorShape(uint16_t shape);
  TensorShape(uint16_t shape1, uint16_t shape2);
  TensorShape(uint16_t shape1, uint16_t shape2, uint16_t shape3);
  TensorShape(uint16_t shape1, uint16_t shape2, uint16_t shape3,
              uint16_t shape4);
  TensorShape(uint16_t shape1, uint16_t shape2, uint16_t shape3,
              uint16_t shape4, uint16_t shape5);  //NT: there's probably a more generalized way to do this
  TensorShape(array<uint16_t, 1> shape);
  TensorShape(array<uint16_t, 2> shape);
  TensorShape(array<uint16_t, 3> shape);
  TensorShape(array<uint16_t, 4> shape);
  TensorShape(array<uint16_t, 5> shape);

  uint16_t operator[](int i) const;
  uint16_t& operator[](int i);
  void update_dims();
  uint16_t get_linear_size() const;
  uint8_t num_dims() const;
  uint32_t linear_index(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const;

 private:
  uint16_t _shape[5];
  uint8_t _num_dims;
};
// Do something to remember current type
enum ttype : uint8_t { i8, u8, i16, u16, i32, u32, flt, undefined };
uint8_t type_size(ttype t);

// Need to figure out way of defering reference until after lefthand assignment
// Need to figure out way of maintaining reference to rvalue
class IntegralValue {
  void* p;
  uint8_t tmp[sizeof(uint32_t)];  // For storing written rvals
  uint8_t num_bytes;  // Need this for writing, need to find way around it

 public:
  // Explicit
  IntegralValue(void* p);
  /*
    IntegralValue(uint8_t u);
    IntegralValue(int8_t u);
    IntegralValue(uint16_t u);
    IntegralValue(int16_t u);
    IntegralValue(uint32_t u);
    IntegralValue(int32_t u);
    IntegralValue(float u);
  */
  IntegralValue(const uint8_t& u);
  IntegralValue(const int8_t& u);
  IntegralValue(const uint16_t& u);
  IntegralValue(const int16_t& u);
  IntegralValue(const uint32_t& u);
  IntegralValue(const int32_t& u);
  IntegralValue(const float& u);

  IntegralValue(uint8_t&& u);
  IntegralValue(int8_t&& u);
  IntegralValue(uint16_t&& u);
  IntegralValue(int16_t&& u);
  IntegralValue(uint32_t&& u);
  IntegralValue(int32_t&& u);
  IntegralValue(float&& u);
  // IntegralValue& operator=(void* _p) { p = _p; }
  IntegralValue(const IntegralValue& that);
  IntegralValue& operator=(const IntegralValue& that);
  IntegralValue& operator=(IntegralValue&& that);

  operator uint8_t() const;
  operator uint8_t&();
  operator int8_t() const;
  operator int8_t&();

  operator uint16_t() const;
  operator uint16_t&();
  operator int16_t() const;
  operator int16_t&();

  operator uint32_t() const;
  operator uint32_t&();
  operator int32_t() const;
  operator int32_t&();

  operator float() const;
  operator float&();
};

#endif
