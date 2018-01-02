#include "log/logged_entry.h"
#include "log/strict_consistent_store-inl.h"

namespace cert_trans {
template class StrictConsistentStore<LoggedEntry>;
}  // namespace cert_trans
