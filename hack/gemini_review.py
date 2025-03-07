import google.generativeai as genai
from google.generativeai import caching
import os
import os.path
from github import Github
from google.cloud import storage
import random
import re
import requests
import json
from time import sleep
import datetime
import tempfile
import subprocess

# Set the maximum number of comments to post on the PR
MAX_COMMENTS = 50
MAX_COMMENT_EXAMPLES=100
GCS_BUCKET = "hackathon-2025-sme-code-review-train"
# files uploaded to GCS:
# 'json_examples/pr_comments_liggit_thockin_deads2k.json'
# 'json_examples/pr_comments_types_go.json' which includes only types.go and validation.go files.
# 'json_examples/pr_comments_all.json'
EXAMPLE_PR_COMMENTS_ALL_BLOB = 'json_examples/pr_comments_all.json'
EXAMPLE_PR_COMMENTS_TYPES_BLOB = 'json_examples/pr_comments_types_go.json'

total_comments_posted = 0
use_context_cache = False
max_prompt_length = 3000000
auto_delete_old_bot_comments = False
print_full_instructions_to_log = False
# can be 'LATEST' 'PATH_TREE_SIMILARITY' 'RANDOM'
sort_historical_prs_by = 'PATH_TREE_SIMILARITY'

robot_name = 'review-bot'


def parse_diff_chunk(diff_chunk):
    lines = diff_chunk.splitlines()
    old_line_num = 0
    new_line_num = 0
    changes = []
    for line in lines:
        header_match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if header_match: 
            old_line_num = int(header_match.group(1))
            new_line_num = int(header_match.group(2))
            continue
        if line.startswith(" "):
            old_line_num += 1
            new_line_num += 1
        elif line.startswith("-"):
            changes.append(f"line {old_line_num}:`{line}`")
            old_line_num += 1
        elif line.startswith("+"):
            changes.append(f"line {new_line_num}:`{line}`")
            new_line_num += 1
    return "<diff-chunk>\n" + diff_chunk + "\n</diff-chunk>\n<changes>\n" + '\n'.join(changes)+"</changes>\n"


def should_skip_file(filename):
    if not filename:
        return True
    # if not (filename.endswith("types.go") or filename.endswith("validation.go")):
    #     return True
    if not filename.endswith(".go"):
        return True
    if filename.endswith("_test.go") or filename.endswith(".pb.go"):
        return True
    if "/test/" in filename:
        return True
    if "/vendor/" in filename:
        return True
    if "_generated" in filename:
        return True
    return False

def sort_historical_prs(historical_prs, pr, mode='LATEST'):
    if mode == 'LATEST':
        historical_prs.sort(key=lambda x: x['created_at'], reverse=True)
    elif mode == 'RANDOM':
        random.shuffle(historical_prs)
    elif mode == 'PATH_TREE_SIMILARITY':
        for historical_pr in historical_prs:
            historical_pr['similarity'] = compute_path_tree_similarities(historical_pr['file_paths'], pr['file_paths'])
            print(f"PR {historical_pr['url']} has similarity {historical_pr['similarity']} with {pr['number']}")
        historical_prs.sort(key=lambda x: x['similarity'], reverse=True)
        print("Top3 similar PRs:", [x['url'] for x in historical_prs[:3]])
    return historical_prs

def compute_path_tree_similarities(file_paths1, file_paths2):
    set1 = set()
    set2 = set()
    for path in file_paths1:
        set1.add(path)
        for pos in substring_positions(path, "/"):
            set1.add(path[:pos])
    for path in file_paths2:
        set2.add(path)
        for pos in substring_positions(path, "/"):
            set2.add(path[:pos])
    return len(set1.intersection(set2))

def substring_positions(s, substring):
    return [i for i in range(len(s)) if s.startswith(substring, i)]

def get_pr_details(repo_name, pr_number, github_token):
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    result = {'title': pr.title, 'number': pr.number, 'url': pr.url, 'body': pr.body, 'author': pr.user.login, 'commits': []}
    result['files'] = get_pr_files(repo_name, pr_number, github_token)
    try:
        commits = list(pr.get_commits())
        for commit in commits:
            commit_data = {'commit_id': commit.sha, 'commit_url': commit.html_url}
            result['commits'].append(commit_data)
    except Exception as e:
        print(f"Error getting commits: {e}")
        return None
    return result

def get_pr_files(repo, pr_number, github_token):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
    headers = github_headers(github_token)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    data_cols_to_keep = ['filename', 'patch', 'status', 'sha', 'raw_url']
    result = []
    for f in data:
        if not should_skip_file(f['filename']):
            result.append({k: v for k, v in f.items() if k in data_cols_to_keep})
    return result

def get_raw_data(url, github_token):
    try:
        headers = github_headers(github_token)
        url = url.replace("https://github.com/gke-ai-hackathon/kubernetes/raw", f"https://raw.githubusercontent.com/gke-ai-hackathon/kubernetes")
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        sleep(0.1)
        return response.text
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

# get the file content before applying the patch    
def get_original_file_content(f, github_token):
    url = f['raw_url']
    content = get_raw_data(url, github_token)
    if content:
        return reverse_patch(content, f['patch'])
    return None

def reverse_patch(modified_txt, diff_txt):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write(modified_txt)
        temp_file_path = temp_file.name
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as diff_file:
        diff_file.write(diff_txt)
        diff_file_path = diff_file.name 
    try:
        result = subprocess.run(['patch', '-R', temp_file_path, diff_file_path], capture_output=True, text=True, check=True)
        print(f"Subprocess output: {result.stdout}")
        with open(temp_file_path, "r") as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
    finally:
        print("Cleaning up temporary files...")
        print(temp_file_path)
        print(diff_file_path)
        os.remove(temp_file_path)
        os.remove(diff_file_path)
    return None

def get_commit_diff_files(commit):
    diff_files = []
    for file in commit.files:
        if not should_skip_file(file.filename):
            if file.patch:
                diff_files.append({
                    'filename': file.filename,
                    'patch': file.patch
                })
    return diff_files

def download_and_combine_guidelines(bucket_name, prefix):
    """Downloads markdown files from GCS using the google-cloud-storage library."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)  # Use prefix for efficiency

        guidelines_content = ''
        for blob in blobs:
            if blob.name.endswith(".md"):
                guidelines_content += blob.download_as_text() + "\n\n"
        return guidelines_content

    except Exception as e:
        print(f"Error downloading or combining guidelines: {e}")

def download_gcs_file_as_json(bucket_name, source_blob_name):
    print(f"trying to download GCS: gs://{bucket_name}/{source_blob_name} as json")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        if not blob.exists():
            print(f"GCS blob does not exist {source_blob_name}")
            return []
        data = json.loads(blob.download_as_string())
        print(f"Successfully downloaded json from GCS {source_blob_name}")
        return data
    except Exception as e:
        print(f"Error downloading file {source_blob_name} from GCS: {e}")
        return []

def get_current_api_definition(repo_name, file_path, github_token):
    """Retrieves the current API definition from given file in the repository."""
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    try:
        file_content = repo.get_contents(file_path)
        return file_content.decoded_content.decode()
    except Exception as e:
        print(f"Error getting file content: {e}")
        return None
    
def comment_prompt_template(comment, is_example=True):
    if not is_example:
        file = comment
        diff = parse_diff_chunk(file['patch'])
        return f"""
Input:
Review the following diff of {file['filename']}:
{diff}
Comments in json format:
"""

    diff = parse_diff_chunk(comment['diff_hunk'])
    line_num = comment['line'] if 'line' in comment and comment['line'] else comment['original_line']
    if not diff:
        return None
    return f"""
Input:
Review the following diff of {comment['path']}:
{diff}
Comments in json format:

Output:
```
[
    {{
        "line": {line_num},
        "comment": "{comment['body']}"
    }}
]
```

"""

def pr_prompt_template(pr, github_token):
    files_content = []
    for file in pr['files']:
        content = get_raw_data(file['raw_url'], github_token)
        if content:
            files_content.append(f"{file['filename']}: \n```\n" + content+"\n```\n")
    files_content_concat = "\n".join(files_content)
    return f"""
Now this is pull request you are reviewing:
<pr-context>
Title: {pr['title']}
Description: {pr['body']}
Files Changed: 
{files_content_concat}
</pr-context>
"""

def generate_instructions(guidelines):    
    instructions = f"""
    You are an expert Kubernetes API reviewer, your task is to review a pull request written in the go programming language.

    Follow the following guidelines written in markdown language:
    {guidelines}

    Review the following pull request. 

    Your task is to identify potential issues and suggest concrete improvements. 

    Prioritize comments that highlight potential bugs, suggest improvements. 
    In your feedback, focus on the types.go files and validation files and functions. 
    Make sure the API changes follow the API conventions. Any changes to existing APIs should be backward compatible.

    Avoid general comments that simply acknowledge correct code or good practices.

* **Adhere to Conventions:**
    * Duration fields use `fooSeconds`.
    * Condition types are `PascalCase`.
    * Constants are `CamelCase`.
    * No unsigned integers.
    * Floating-point values are avoided in `spec`.
    * Use `int32` unless `int64` is necessary.
    * `Reason` is a one-word, `CamelCase` category of cause.
    * `Message` is a human-readable phrase with specifics.
    * Label keys are lowercase with dashes.
    * Annotations are for tooling and extensions.
* **Compatibility:**
    * Added fields must have non-nil default values in all API versions.
    * New enum values must be handled safely by older clients.
    * Validation rules on spec fields cannot be relaxed nor strengthened.
    * Changes must be round-trippable with no loss of information.
* **Changes:**
    * New fields should be optional and added in a new API version if possible.
    * Singular fields should not be made plural without careful consideration of compatibility.
    * Avoid renaming fields within the same API version.
    * When adding new fields or enum values, use feature gates to control enablement and ensure compatibility with older API servers.

When reviewing a pull request, you will be given first the overall context of the pull request enclosed in the pr-context section.
Then you will be asked to review a specific file, with
1. the file diff chunk enclosed in the diff-chunk section
3. lines changed enclosed in the changes section in the form of line number: line diff

You should 
* consider all the pull request context provided in the pr-context section
* focus on the diff-chunk of the file requested
* review the lines changed in the changes section line by line, and provide feedback for a line if there is anything concerning.
You reply should only contain the feedback comments. And they should be in the following json format, referencing the line number of the line change the feedback is about:
```
[
    {{
        "line": <line_number>,
        "comment": "<comment>"
    }},
    {{
        "line": <line_number>,
        "comment": "<comment>"
    }},
    ...and so on
]
```

    """
    return instructions

def match_comment_to_file(file_to_review, comment):
    filename = file_to_review['filename'].split("/")[-1]
    return filename in comment['path']

def add_examples_to_prompt(file_to_review, pr_comments, max_examples):  
    if max_examples == 0 or len(pr_comments) == 0:
        return ""
    instructions = ""
    matching_comments = []
    other_comments = []
    for pr in pr_comments:
        for comment in pr['comments']:
            if match_comment_to_file(file_to_review, comment):
                matching_comments.append(comment_prompt_template(comment))
            else:
                other_comments.append(comment_prompt_template(comment))
    selected_comments = matching_comments + other_comments
    if max_examples < len(selected_comments):
        selected_comments = selected_comments[:max_examples]
    instructions += "Here are some examples:\n"
    num_examples = 0
    for comment in selected_comments:
        example_prompt = "<example>\n" + comment + "</example>\n\n"
        if len(example_prompt) >= max_prompt_length - len(instructions):
            break
        instructions += example_prompt
        num_examples += 1
    print(f"DEBUG: Added {num_examples} examples out of {len(matching_comments)} matching ones to the instructions prompt.")
    return instructions

def generate_gemini_review_for_single_file(repo, pr, file, github_token, api_key, guidelines, pr_comments_all, max_examples=20, print_prompt=False):
    """Generates a code review with annotations using Gemini."""
    if total_comments_posted >= MAX_COMMENTS:
        return
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    print(f"*********Generating review for {file['filename']}***********")

    # Add guidelines.
    prompt = generate_instructions(guidelines)
    # Add example PR reviews.
    prompt += add_examples_to_prompt(file, pr_comments_all, max_examples)
    
    prompt += pr_prompt_template(pr, github_token)

    file_content = get_original_file_content(file, github_token)
    if not file_content:
        return
    prompt += f"""
Carefully review the proposed changes in the context of PR above and the full existing file content provided below:
{file_content}

{comment_prompt_template(file, is_example=False)}
"""
    response = model.generate_content(prompt)
    if print_prompt:
        print("DEBUG me: ", prompt)
    if not response:
        print("DEBUG gemini: Empty")
        return
    print("DEBUG gemini: ", response.text)
    print(response.usage_metadata)
    gemini_comments = parse_gemini_response(response.text)
    if not gemini_comments:
        return
    for comment in gemini_comments:
        comment['path'] = file['filename']
        comment['commit_id'] = pr['commits'][-1]['commit_id']
        post_github_comment(repo, pr['number'], github_token, comment)
        if total_comments_posted >= MAX_COMMENTS:
            break
        
def generate_diff_prompt(repo, pr, github_token, diff_file, max_examples=20):
    """Generates a prompt for the Gemini model."""
    max_diff_length = 100000
    diff = parse_diff_chunk(diff_file['patch'])
    if len(diff) > max_diff_length:
        diff = diff[:max_diff_length] + "\n... (truncated due to length limit)..."

    diff_prompt = "" 
    if "types.go" in diff_file['filename'] or "validation.go" in diff_file['filename']:
        current_api_definition = get_current_api_definition(repo, diff_file['filename'], github_token)
        if current_api_definition:
            diff_prompt +=  prompt_for_types(repo, github_token, diff_file, max_examples=max_examples)   

    diff_prompt += f"""
Now review the following file in pull request #{pr['number']} (Remember, do not comment on any line that is not in the changes section):
file path:{diff_file['filename']}
{diff}
"""
    diff_prompt +=  prompt_reminder() + f"""
    Comments in json format:
    """
    return diff_prompt

def prompt_reminder():
    return """
    Remember: You are a code reviewer. Your job is to find bugs, errors, and omissions in the code. 
    Do not praise the author. 
    Do not make any conversational comments. Focus only on code quality.
    Please avoid general comments that simply acknowledge correct code or good practices. 
    Instead, focus on providing actionable feedback that can help improve the code's quality and robustness. 
    Only leave a review comment if it directly relates to one of the guidelines above. 
    Avoid making general comments or observations that are not relevant to these specific guidelines.
    Do not comment on the existing comments in the code.
    Important- Prioritize technical accuracy over conversational language.
    """

def prompt_for_types(repo, github_token, diff_file, max_examples=20):
    pr_comments_types = download_gcs_file_as_json(GCS_BUCKET, EXAMPLE_PR_COMMENTS_TYPES_BLOB)
    print(f"downloaded {len(pr_comments_types)} example pr (types.go) comments")
    types_prompt = ""
    current_api_definition = get_current_api_definition(repo, diff_file['filename'], github_token)
    if current_api_definition:
        types_prompt +=  f"""
Carefully review the proposed changes in the context of the full existing API definition provided below:
{current_api_definition}
"""
    types_prompt += add_examples_to_prompt(pr_comments_types, max_examples)
    print("finished adding types specific prompt")
    return types_prompt
    
    
def send_prompt_to_gemini(chat, prompt):
    print("me:", prompt)
    response = chat.send_message(prompt)
    print("gemini:", response.text)
    print("******end of gemini response********")
    print(response.usage_metadata)
    return response

# parse gemini response into json object
# expect the response to be like
# ```json
# [
#     {
#         "line": 451,
#         "comment": "little bunny foo foo"
#     }
# ]
# ```
def parse_gemini_response(r):
    r = r.strip()
    begin = r.find("```json")
    if begin < 0:
        print("Error: Gemini response does not start with ```json ", r)
        return None
    r = r[begin:]
    r = r.removeprefix("```json")
    if r.endswith("```"):
        r = r.removesuffix("```")
    else:
        end = r.find("```")
        if end < 0:
            print("Error: Gemini response does not end with ```", r)
            return None
        r = r[:end]
    r = r.strip().replace("\n", " ")
    try:
        ret = json.loads(r)
        if not isinstance(ret, list):
            print("Error: Gemini response is not a list - ", ret)
            return None
        print("successfully parsed gemini response")
        return ret
    except Exception as e:
        print("Error: failed to parse gemini response: ", e)
        print(r)
        return None

def github_headers(github_token):
    return {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

def post_github_comment_reply(repo, pr_number, github_token, reply_to, content):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments/{reply_to}/replies"
    print(f"post_github_comment_reply to {url}")
    headers = github_headers(github_token)
    payload = {
        "body": content,
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        data = response.json()
        if 'url' in data:
            print(f"posted comment reply to {reply_to}", data['url'])
            print(json.dumps(data, indent=2))
    else:
        print("failed to post comment reply: ", response.text)
        print("payload:", payload)
    # wait some time before posting next comment to avoid rate limiting error.
    sleep(0.3)


def post_github_comment(repo, pr_number, github_token, comment):
    global total_comments_posted
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    headers = github_headers(github_token)
    payload = {
        "body": comment['comment'],
        "commit_id": comment['commit_id'],
        "line": comment['line'],
        "path": comment['path'],
        "side": "RIGHT",
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        data = response.json()
        if 'url' in data:
            print("posted comment ", data['url'])
        total_comments_posted = total_comments_posted + 1
    else:
        print("failed to post comment: ", response.text)
        print("payload:", payload)
    # wait some time before posting next comment to avoid rate limiting error.
    sleep(0.3)

def get_pr_comments(repo, pr_number, github_token):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    comment_id_to_comment = {}
    page = 1
    headers = github_headers(github_token)
    comment_keys_to_keep = [
        'url', 'in_reply_to_id', 'id', 'created_at', 'updated_at', 'diff_hunk', 'body', 'commit_id', 'original_commit_id', 
        'path', 'line', 'start_line', 'side', 'start_side', 'original_line', 'original_start_line', 'position', 'original_position']
    while True:
        response = requests.get(url, headers=headers, params={"per_page": 100, "page": page})
        response.raise_for_status()
        data = response.json()
        for comment_data in data:
            comment = {'commenter': comment_data['user']['login']}
            for key in comment_keys_to_keep:
                if key in comment_data:
                    comment[key] = comment_data[key]
            comment_id_to_comment[comment_data['id']] = comment
        if len(data) < 100:
            break
        page += 1
    print(f"{url}\ntotal comments: {len(comment_id_to_comment)}")
    # print(comment_id_to_comment)
    return comment_id_to_comment

def delete_old_bot_comments(repo, pr_number, github_token):
    comment_id_to_comment = get_pr_comments(repo, pr_number, github_token)
    headers = github_headers(github_token)
    total_comments_deleted = 0
    
    for id, comment in comment_id_to_comment.items():
        if comment['commenter'] == 'github-actions[bot]':
            comment_url = f"https://api.github.com/repos/{repo}/pulls/comments/{id}"
            response = requests.delete(comment_url, headers=headers)
            if response.status_code == 204:
                total_comments_deleted += 1
            else:
                print(f"failed to delete comment {id}: {response.text}")
            sleep(0.1)
    print(f"deleted {total_comments_deleted} old bot comments")

def get_comments_needing_reply(comment_id_to_comment):
    result = []
    print(json.dumps(comment_id_to_comment, indent=2))
    # the 'in_reply_to_id' of a reply comment seems always pointing to the root of the list instead of previous comment.
    threads = {}
    for id, comment in comment_id_to_comment.items():
        root_id = comment['id']
        if 'in_reply_to_id' in comment and comment['in_reply_to_id']:
            root_id = comment['in_reply_to_id']
        if root_id not in threads:
            threads[root_id] = []
        threads[root_id].append(comment)
        
    for root_id, comments in threads.items():
        comments.sort(key=lambda x: x['created_at'], reverse=True)
        print(f"comments in thread {root_id}: {json.dumps(comments, indent=2)}")
        idx_needs_reply = -1
        for idx, comment in enumerate(comments):
            if comment['commenter'] == 'github-actions[bot]':
                break
            if f"@{robot_name}" in comment['body']:
                idx_needs_reply = idx
                break
        if idx_needs_reply < 0:
            continue
        chain = comments[idx_needs_reply:]
        result.append(chain)
    print(f"threads awaiting reply: {json.dumps(result, indent=2)}")
    return result


def generate_gemini_reply_for_comment(pr, file, github_token, api_key, guidelines, conversation, print_prompt=False):
    """Generates a code review with annotations using Gemini."""
    if total_comments_posted >= MAX_COMMENTS:
        return
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    print(f"*********Generating review for {file['filename']}***********")

    # Add guidelines.
    prompt = f"""
    You are an expert Kubernetes API reviewer, your task is to review a pull request written in the go programming language and respond to review comments.

    Any API change should follow the following guidelines written in markdown language:
    {guidelines}

    You are currently reviewing the following pull request:
    """
    prompt += pr_prompt_template(pr, github_token)

    file_content = get_original_file_content(file, github_token)
    if not file_content:
        return
    diff = parse_diff_chunk(file['patch'])
    prompt += f"""
You are currently looking at the {file['filename']} file this PR is changing: 
{file_content}

You have been asked to provide feedbacks for the following diff for the file above:
{diff}

And this is the conversation that has happened so far (You are {robot_name} in this conversation). 
{conversation}

Now please reply to the last comment in this conversation:
"""
    response = model.generate_content(prompt)
    if print_prompt:
        print("DEBUG me: ", prompt)
    else:
        print(prompt[-1000:])
    if not response:
        print("DEBUG gemini: Empty")
        return None
    print("DEBUG gemini: ", response.text)
    print(response.usage_metadata)
    return response.text

def reply_to_comments(comment_id_to_comment, repo, pr, api_key, guidelines, github_token):
    path_to_file = {}
    for file in pr['files']:
        path_to_file[file['filename']] = file
    comments_needing_reply = get_comments_needing_reply(comment_id_to_comment)
    for comment_chain in comments_needing_reply:
        comment_chain.reverse()
        last_comment = comment_chain[-1]
        file = path_to_file[last_comment['path']]
        if not file:
            continue
        conversation = []
        for comment in comment_chain:
            commenter = comment['commenter']
            if commenter == 'github-actions[bot]':
                commenter = robot_name
            conversation.append(f"{commenter}: {comment['body']}")
        conversation_str = "\n".join(conversation)
        reply = generate_gemini_reply_for_comment(pr, file, github_token, api_key, guidelines, conversation_str, print_prompt=False)
        print(f"gemini chat to {last_comment['id']}: {reply}")
        print(f"last comment: {last_comment['body']}")
        post_github_comment_reply(repo, pr['number'], github_token, last_comment['id'], reply)


def post_github_review_comments(repo_name, pr_number, review_comments, github_token):
    """Posts review comments to GitHub PR, annotating specific lines."""
    global total_comments_posted  # Declare total_comments_posted as global
    if not review_comments:
        return
    for review_comment in review_comments:
        if total_comments_posted >= MAX_COMMENTS:
            return
        post_github_comment(repo_name, pr_number, github_token, review_comment)
        print(f"Review comments for {review_comment['path']} posted.")

def get_latest_bot_comment_timestamp(repo_name, pr_number, github_token):
    """Gets the timestamp of the latest bot comment on the PR using PyGithub."""
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    
    latest_bot_comment_time = None
    
    # Check review comments (line-specific)
    for comment in pr.get_review_comments():
        if comment.user.login == 'github-actions[bot]':
            comment_time = comment.created_at
            if latest_bot_comment_time is None or comment_time > latest_bot_comment_time:
                latest_bot_comment_time = comment_time
    
    # Also check issue comments (general PR comments)
    for comment in pr.get_issue_comments():
        if comment.user.login == 'github-actions[bot]':
            comment_time = comment.created_at
            if latest_bot_comment_time is None or comment_time > latest_bot_comment_time:
                latest_bot_comment_time = comment_time
    
    return latest_bot_comment_time

def main():
    """Main function to orchestrate Gemini PR review."""
    api_key = os.environ.get('GEMINI_API_KEY')
    pr_number = int(os.environ.get('PR_NUMBER'))
    repo_name = os.environ.get('GITHUB_REPOSITORY')
    github_token = os.environ.get('GITHUB_TOKEN')

    pr = get_pr_details(repo_name, pr_number, github_token)
    if not pr:
        print("Could not retrieve PR details. Exiting.")
        return
        
    # Check if there are files to review
    if not pr.get('files') or len(pr['files']) == 0:
        print("No files to review after filtering. Skipping review.")
        return
    pr['file_paths'] = [file['filename'] for file in pr['files']]

    guidelines = download_and_combine_guidelines(GCS_BUCKET, "guidelines/")
    if not guidelines:
        print("Warning: No guidelines loaded.")
    
    if auto_delete_old_bot_comments:
        delete_old_bot_comments(repo_name,pr_number,github_token)
    else :
        existing_comments = get_pr_comments(repo_name, pr_number, github_token)
        reply_to_comments(existing_comments, repo_name, pr, api_key, guidelines, github_token)
        if len(existing_comments) > 0:
            return
    
    # Get the latest commit timestamp
    if pr['commits'] and len(pr['commits']) > 0:
        # Get the commit details to extract timestamp
        latest_commit_id = pr['commits'][-1]['commit_id']
        g = Github(github_token)
        repo = g.get_repo(repo_name)
        commit = repo.get_commit(latest_commit_id)
        latest_commit_time = commit.commit.author.date
        
        # Get the latest bot comment timestamp
        latest_bot_comment_time = get_latest_bot_comment_timestamp(repo_name, pr_number, github_token)
        
        # If the bot has already commented after the latest commit, skip the review
        if latest_bot_comment_time and latest_bot_comment_time > latest_commit_time:
            print(f"Latest bot comment ({latest_bot_comment_time}) is newer than the latest commit ({latest_commit_time}). Skipping review.")
            return
        
        print(f"Latest commit: {latest_commit_time}")
        if latest_bot_comment_time:
            print(f"Latest bot comment: {latest_bot_comment_time}")
        else:
            print("No previous bot comments found.")
    
    pr_comments_all = download_gcs_file_as_json(GCS_BUCKET, EXAMPLE_PR_COMMENTS_ALL_BLOB)
    print(f"downloaded {len(pr_comments_all)} example pr (all) comments")
    for pr_comment in pr_comments_all:
        pr_comment['file_paths'] = ['' if should_skip_file(file['filename']) else file['filename'] for file in pr_comment['files']]
    pr_comments_all = sort_historical_prs(pr_comments_all, pr, mode=sort_historical_prs_by)

    print_prompt = True
    for file in pr['files']:
        if not should_skip_file(file['filename']):
            generate_gemini_review_for_single_file(repo_name, pr, file, github_token, api_key, guidelines, pr_comments_all, max_examples=MAX_COMMENT_EXAMPLES, print_prompt=print_prompt)
            print_prompt = False
    print(f"total_comments_posted = {total_comments_posted}")

if __name__ == "__main__":
    main()
