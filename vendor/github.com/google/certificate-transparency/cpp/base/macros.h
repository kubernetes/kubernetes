#ifndef CERT_TRANS_BASE_MACROS_H_
#define CERT_TRANS_BASE_MACROS_H_


// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete


#endif  // CERT_TRANS_BASE_MACROS_H_
