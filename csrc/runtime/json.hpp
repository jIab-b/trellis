#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

namespace trellis::json {

struct Value;
using Object = std::unordered_map<std::string, Value>;
using Array  = std::vector<Value>;

struct Value {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type{Type::Null};
  bool b{false};
  double num{0};
  std::string s;
  Array a;
  Object o;

  static Value null() { return Value{}; }
  static Value boolean(bool v) { Value x; x.type=Type::Bool; x.b=v; return x; }
  static Value number(double v) { Value x; x.type=Type::Number; x.num=v; return x; }
  static Value string(std::string v) { Value x; x.type=Type::String; x.s=std::move(v); return x; }
  static Value array(Array v) { Value x; x.type=Type::Array; x.a=std::move(v); return x; }
  static Value object(Object v) { Value x; x.type=Type::Object; x.o=std::move(v); return x; }

  bool is_object() const { return type==Type::Object; }
  bool is_array() const { return type==Type::Array; }
  bool is_string() const { return type==Type::String; }
  bool is_number() const { return type==Type::Number; }
  bool is_bool() const { return type==Type::Bool; }
};

Value parse(const std::string& s);

// helpers
inline const Value* get(const Object& o, const std::string& k) {
  auto it = o.find(k);
  return it == o.end() ? nullptr : &it->second;
}

} // namespace trellis::json

