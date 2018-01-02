for selfsigned in delgkey1 delgkey2 delgkey3 delgkey4; do
        subj='/C=US/ST=CA/L=SanFrancisco/O=Docker/CN=delegation'

        openssl genrsa -out "${selfsigned}.key" 2048
        openssl req -new -key "${selfsigned}.key" -out "${selfsigned}.csr" -sha256 -subj "${subj}"
        cat > "${selfsigned}.cnf" <<EOL
[selfsigned]
basicConstraints = critical,CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage=codeSigning
subjectKeyIdentifier=hash
EOL

        openssl x509 -req -days 3560 -in "${selfsigned}.csr" -signkey "${selfsigned}.key" -sha256 \
                -out "${selfsigned}.crt" -extfile "${selfsigned}.cnf" -extensions selfsigned

        rm "${selfsigned}.cnf" "${selfsigned}.csr"
done
