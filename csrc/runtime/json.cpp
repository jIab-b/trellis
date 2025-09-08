#include "json.hpp"
#include <cctype>
#include <stdexcept>

namespace trellis::json {

struct Parser {
  const std::string& s;
  size_t i{0};
  Parser(const std::string& s) : s(s) {}

  void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }

  bool match(char c) { ws(); if (i<s.size() && s[i]==c) { ++i; return true; } return false; }
  void expect(char c) { ws(); if (i>=s.size() || s[i]!=c) throw std::runtime_error("json: expected '"+std::string(1,c)+"'"); ++i; }

  Value parse_value() {
    ws(); if (i>=s.size()) throw std::runtime_error("json: unexpected end");
    char c = s[i];
    if (c=='{') return parse_object();
    if (c=='[') return parse_array();
    if (c=='"') return parse_string();
    if (c=='t' || c=='f') return parse_bool();
    if (c=='n') return parse_null();
    return parse_number();
  }

  Value parse_object() {
    expect('{');
    Object o;
    ws();
    if (match('}')) return Value::object(std::move(o));
    while (true) {
      ws();
      std::string key = parse_string().s;
      expect(':');
      Value val = parse_value();
      o.emplace(std::move(key), std::move(val));
      ws();
      if (match('}')) break;
      if (!match(',')) throw std::runtime_error("json: expected ',' or '}'");
    }
    return Value::object(std::move(o));
  }

  Value parse_array() {
    expect('[');
    Array a;
    ws();
    if (match(']')) return Value::array(std::move(a));
    while (true) {
      a.push_back(parse_value());
      ws();
      if (match(']')) break;
      if (!match(',')) throw std::runtime_error("json: expected ',' or ']'");
    }
    return Value::array(std::move(a));
  }

  Value parse_string() {
    expect('"');
    std::string out;
    while (i < s.size()) {
      char c = s[i++];
      if (c=='"') break;
      if (c=='\\') {
        if (i>=s.size()) throw std::runtime_error("json: bad escape");
        char e = s[i++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            // minimal \uXXXX handling for BMP range
            if (i+4> s.size()) throw std::runtime_error("json: short unicode escape");
            unsigned code=0; for(int k=0;k<4;++k){ char h=s[i++]; code <<= 4; if(h>='0'&&h<='9') code+=h-'0'; else if(h>='a'&&h<='f') code+=h-'a'+10; else if(h>='A'&&h<='F') code+=h-'A'+10; else throw std::runtime_error("json: bad hex"); }
            if (code<=0x7F) out.push_back(static_cast<char>(code));
            else if (code<=0x7FF) { out.push_back(static_cast<char>(0xC0 | ((code>>6)&0x1F))); out.push_back(static_cast<char>(0x80 | (code&0x3F))); }
            else { out.push_back(static_cast<char>(0xE0 | ((code>>12)&0x0F))); out.push_back(static_cast<char>(0x80 | ((code>>6)&0x3F))); out.push_back(static_cast<char>(0x80 | (code&0x3F))); }
            break;
          }
          default: throw std::runtime_error("json: bad escape char");
        }
      } else {
        out.push_back(c);
      }
    }
    return Value::string(std::move(out));
  }

  Value parse_bool() {
    if (s.compare(i, 4, "true")==0) { i+=4; return Value::boolean(true); }
    if (s.compare(i, 5, "false")==0) { i+=5; return Value::boolean(false); }
    throw std::runtime_error("json: invalid bool");
  }

  Value parse_null() {
    if (s.compare(i, 4, "null")==0) { i+=4; return Value::null(); }
    throw std::runtime_error("json: invalid null");
  }

  Value parse_number() {
    size_t start = i;
    if (s[i]=='-') ++i;
    while (i<s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    if (i<s.size() && s[i]=='.') { ++i; while (i<s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i; }
    if (i<s.size() && (s[i]=='e' || s[i]=='E')) { ++i; if (s[i]=='+'||s[i]=='-') ++i; while (i<s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i; }
    double v = std::stod(s.substr(start, i-start));
    return Value::number(v);
  }
};

Value parse(const std::string& s) {
  Parser p(s);
  Value v = p.parse_value();
  p.ws();
  if (p.i != s.size()) throw std::runtime_error("json: trailing characters");
  return v;
}

} // namespace trellis::json

