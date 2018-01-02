from ct.serialization import tls_message
from ct.proto import client_pb2

def decode_entry(entry):
    parsed_entry = client_pb2.ParsedEntry()
    tls_message.decode(entry.leaf_input, parsed_entry.merkle_leaf)

    parsed_entry.extra_data.entry_type = (parsed_entry.merkle_leaf.
                                          timestamped_entry.entry_type)

    tls_message.decode(entry.extra_data, parsed_entry.extra_data)
    return parsed_entry
