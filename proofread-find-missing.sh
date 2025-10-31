#!/usr/bin/env bash
set -euo pipefail

# proofread-find-missing.sh
# Usage: proofread-find-missing.sh [-v] [-s PERCENT] [-p PATTERN] DIRECTORY
# -v : verbose (prints matching filenames that passed checks)
# -s : percent size variance (integer >0 and <50), default 10
# -p : glob pattern to match files in each subdirectory (default '*-proofread.hocr')

VERBOSE=0
S_PERCENT=10
PATTERN='*-proofread.hocr'

usage() {
  cat <<EOF
Usage: $0 [-v] [-s PERCENT] [-p PATTERN] DIRECTORY
  -v            Verbose: print matching filenames that pass the checks (default: off)
  -s PERCENT    Percent size variance (integer >0 and <50). Default: 10
  -p PATTERN    Glob pattern to match files (default: '*-proofread.hocr')
EOF
  exit 2
}

# parse options
while getopts ":vs:p:" opt; do
  case "$opt" in
    v) VERBOSE=1 ;;
    s) S_PERCENT="$OPTARG" ;;
    p) PATTERN="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))

if [ $# -ne 1 ]; then
  usage
fi

TARGET_DIR="$1"

# validate directory
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: '$TARGET_DIR' is not a directory." 1>&2
  exit 1
fi

# validate S_PERCENT is integer >0 and <50
if ! [[ "$S_PERCENT" =~ ^[0-9]+$ ]]; then
  echo "Error: -s value must be an integer." 1>&2
  exit 2
fi

if [ "$S_PERCENT" -le 0 ] || [ "$S_PERCENT" -ge 50 ]; then
  echo "Error: -s value must be greater than 0 and less than 50." 1>&2
  exit 2
fi

# count immediate subdirectories
count=0
shopt -s nullglob
for d in "$TARGET_DIR"/*/; do
  [ -d "$d" ] && count=$((count+1))
done

echo "Checking $count subdirectories in: $TARGET_DIR"

# iterate immediate subdirectories
for sub in "$TARGET_DIR"/*/; do
  [ -d "$sub" ] || continue
  # strip trailing slash for nicer output
  sub_name=$(basename "${sub%/}")

  # find files matching the provided glob PATTERN in this subdirectory
  # default PATTERN is '*-proofread.hocr'
  matches=()
  for f in "$sub"$PATTERN; do
    [ -e "$f" ] || continue
    matches+=("$f")
  done

  if [ ${#matches[@]} -eq 0 ]; then
    echo "$sub_name: no matching file"
    continue
  fi

  # process each matching file (handles multiple matches)
  for match in "${matches[@]}"; do
    # ensure file still exists (in case of nullglob oddities)
    [ -e "$match" ] || continue

    # check file is not zero size
    if [ ! -s "$match" ]; then
      echo "$sub_name: $(basename "$match") is 0 size"
      continue
    fi

    # derive original filename by removing the first occurrence of '-proofread'
    base_name=$(basename "$match")
    # use bash parameter expansion (pattern-based) to remove the first occurrence
    file_orig="${base_name/-proofread/}"
    orig_path="$sub${file_orig}"

    # if original does not exist
    if [ ! -e "$orig_path" ]; then
      echo "Warning: $sub_name: Original file: $file_orig does not exist."
      continue
    fi

    # get sizes
    # use stat -c %s for portability on GNU; fall back to wc -c on systems without it
    if stat_out=$(stat -c %s -- "$match" 2>/dev/null); then
      proof_size=$stat_out
    else
      proof_size=$(wc -c <"$match" | tr -d '[:space:]')
    fi

    if stat_out=$(stat -c %s -- "$orig_path" 2>/dev/null); then
      orig_size=$stat_out
    else
      orig_size=$(wc -c <"$orig_path" | tr -d '[:space:]')
    fi

    # if original is zero size, warn and skip percent comparison
    if [ "$orig_size" -eq 0 ]; then
      echo "Warning: $sub_name: Original file: $file_orig is 0 size"
      continue
    fi

    # compute allowed bounds: lower = orig_size * (1 - S_PERCENT/100), upper = orig_size * (1 + S_PERCENT/100)
    # use awk for float math and convert to integer bounds
    lower=$(awk -v s="$orig_size" -v p="$S_PERCENT" 'BEGIN{printf("%d", s*(1 - p/100))}')
    upper=$(awk -v s="$orig_size" -v p="$S_PERCENT" 'BEGIN{printf("%d", s*(1 + p/100))}')

    # Ensure bounds at least 1
    [ "$lower" -lt 1 ] && lower=1

    if [ "$proof_size" -lt "$lower" ] || [ "$proof_size" -gt "$upper" ]; then
      echo "$sub_name: $(basename "$match") size $proof_size not within $S_PERCENT% of $file_orig size $orig_size"
      continue
    fi

    # If all checks passed, print matching filename only if verbose
    if [ "$VERBOSE" -eq 1 ]; then
      echo "$match"
    fi
  done
done
