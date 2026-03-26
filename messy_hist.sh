# create fake messy commits
for i in {1..10}; do echo $i >> file.txt; git add .; git commit -m "temp $i"; done

# create branch conflicts
git checkout -b feature-x
echo "feature change" >> file.txt
git commit -am "feature work"

git checkout main
echo "main change" >> file.txt
git commit -am "main work"
