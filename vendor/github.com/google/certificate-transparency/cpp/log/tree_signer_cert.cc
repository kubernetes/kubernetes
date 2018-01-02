#include "log/logged_entry.h"
#include "log/tree_signer-inl.h"

namespace cert_trans {
template class TreeSigner<cert_trans::LoggedEntry>;
}  // namespace cert_trans
