#include "log/database.h"

namespace cert_trans {


DatabaseNotifierHelper::~DatabaseNotifierHelper() {
  CHECK(callbacks_.empty());
}


void DatabaseNotifierHelper::Add(const NotifySTHCallback* callback) {
  CHECK(callbacks_.insert(callback).second);
}


void DatabaseNotifierHelper::Remove(const NotifySTHCallback* callback) {
  Map::iterator it(callbacks_.find(callback));
  CHECK(it != callbacks_.end());

  callbacks_.erase(it);
}


void DatabaseNotifierHelper::Call(const ct::SignedTreeHead& sth) const {
  for (Map::const_iterator it = callbacks_.begin(); it != callbacks_.end();
       ++it) {
    (**it)(sth);
  }
}


}  // namespace cert_trans
