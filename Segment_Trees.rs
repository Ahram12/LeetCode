//Leetcode 3165
use std::cmp;

#[derive(Debug)]
struct Node {
    //Range of array represented by node
    range: (usize, usize),
    
    //Maximum sum without first element
    prefix: i64,
    
    //Maximum sum without last element
    suffix: i64,
    
    //Maximum sum without first or last element
    middle: i64,
    
    //Maximum sum that meets condition overall
    max: i64,
    
    //Left child node
    left: Option<Box<Node>>,
    
    //Right child node
    right: Option<Box<Node>>
}

impl Node {
    fn new(range: (usize, usize), max: i64) -> Node {
         return Node {range: range, prefix: 0, suffix: 0, middle: 0, max: max, left: None, right: None }
    }
    
    fn build(arr: &Vec<i32>, range: (usize, usize)) -> Option<Box<Self>> {
        const MOD: i64 = 10_i64.pow(9) + 7;
        
        if range.0 > range.1 {
            return None;
        }
        
        let mid = (range.0 + range.1) / 2;
        
        let mut node = Node::new(range, cmp::max(arr[range.0] as i64, 0));
        
        if range.0 < range.1 {
            node.left = Self::build(arr, (range.0, mid));
            node.right = Self::build(arr, (mid + 1, range.1));
            
            let left_prefix = node.left.as_ref().unwrap().prefix;
            let left_suffix = node.left.as_ref().unwrap().suffix;
            let left_middle = node.left.as_ref().unwrap().middle;
            let left_max = node.left.as_ref().unwrap().max;
                
            let right_prefix = node.right.as_ref().unwrap().prefix;
            let right_suffix = node.right.as_ref().unwrap().suffix;
            let right_middle = node.right.as_ref().unwrap().middle;
            let right_max = node.right.as_ref().unwrap().max;
                
            node.prefix = cmp::max(left_middle + right_max, left_prefix + right_prefix) % MOD;
            node.suffix = cmp::max(left_max + right_middle, left_suffix + right_suffix) % MOD;
            node.middle = cmp::max(left_middle + right_middle, cmp::max(left_prefix + right_middle, left_middle + right_suffix)) % MOD;
            node.max = cmp::max(left_suffix + right_max, left_max + right_prefix) % MOD;
        }
        
        return Some(Box::new(node))
    }

    fn update(&mut self, index: usize, value: i64) {
         if self.range.0 == index && self.range.1 == index {
            self.max = value;
        } else {
            let mid = (self.range.0 + self.range.1)/2;
            if index <= mid {
                self.left.as_mut().unwrap().update(index, value);
            } else {
                self.right.as_mut().unwrap().update(index, value);
            }
            const MOD: i64 = 10_i64.pow(9) + 7;
            let left_prefix = self.left.as_ref().unwrap().prefix;
            let left_suffix = self.left.as_ref().unwrap().suffix;
            let left_middle = self.left.as_ref().unwrap().middle;
            let left_max = self.left.as_ref().unwrap().max;
                
            let right_prefix = self.right.as_ref().unwrap().prefix;
            let right_suffix = self.right.as_ref().unwrap().suffix;
            let right_middle = self.right.as_ref().unwrap().middle;
            let right_max = self.right.as_ref().unwrap().max;
                
            self.prefix = cmp::max(left_middle + right_max, left_prefix + right_prefix) % MOD;
            self.suffix = cmp::max(left_max + right_middle, left_suffix + right_suffix) % MOD;
            self.middle = cmp::max(left_middle + right_middle, cmp::max(left_prefix + right_middle, left_middle + right_suffix)) % MOD;
            self.max = cmp::max(left_suffix + right_max, left_max + right_prefix) % MOD;
        }
    }
}

fn main() {
    let arr: Vec<i32> = vec![3,5,9,4,10];
    let mut tree = Node::build(&arr, (0, arr.len() - 1)).unwrap();
    tree.update(1, -2);
    tree.update(0, 11);
    println!("{:?}", tree)
    
}

//Leetcode 3187

use std::cmp;

#[derive(Debug)]
struct Node {
    //Range of array represented by node
    range: (usize, usize),
    
    //Count of corresponding peaks,
    count: i32,
    
    //Left child node
    left: Option<Box<Node>>,
    
    //Right child node
    right: Option<Box<Node>>
}

impl Node {
    fn new(range: (usize, usize), count: i32) -> Node {
         return Node {range: range, count: count, left: None, right: None }
    }
    
    fn build(arr: &Vec<i32>, range: (usize, usize)) -> Option<Box<Self>> {
        if range.0 > range.1 {
            return None;
        }
        
        let mid = (range.0 + range.1) / 2;
        
        let mut node = Node::new(range, arr[range.0]);
        
        if range.0 < range.1 {
            node.left = Self::build(arr, (range.0, mid));
            node.right = Self::build(arr, (mid + 1, range.1));
            let left_count = node.left.as_ref().unwrap().count;
            let right_count = node.right.as_ref().unwrap().count;
            node.count = left_count + right_count;
        }
        
        return Some(Box::new(node))
    }
    
    fn query(&mut self, range: (usize, usize)) -> i32 {
        if  range.0 > range.1 {
            return 0
        }
        
        if  self.range.0 == range.0 && self.range.1 == range.1 {
            return self.count
        }
        
        let mut count = 0;

        if let Some(ref mut left) = self.left {
            count += left.query((range.0, cmp::min(range.1, left.range.1))); 
        }
        
        
        if let Some(ref mut right) = self.right {
            count += right.query((cmp::max(range.0, right.range.0), range.1));
        }
        
      return count
    }
    
    fn update(&mut self, index: usize, value: i32) {
         if self.range.0 == index && self.range.1 == index {
            self.count = value;
        } else {
            let mid = (self.range.0 + self.range.1)/2;
            if index <= mid {
                self.left.as_mut().unwrap().update(index, value);
            } else {
                self.right.as_mut().unwrap().update(index, value);
            }
            let left_count = self.left.as_ref().unwrap().count;
            let right_count = self.right.as_ref().unwrap().count;
            self.count = left_count + right_count;
        }
    }
}

fn main() {
    let mut arr: Vec<i32> = vec![4,1,4,2,1,5];
    let queries: Vec<Vec<i32>> = vec![vec![2,2,4],vec![1,0,2],vec![1,0,4]];
    let n = arr.len();
    let m = queries.len();
    let mut t: Vec<i32> = vec![0; n];
    
    for i in 1..n - 1 {
        if arr[i - 1] < arr[i] && arr[i] > arr[i + 1] {
            t[i] = 1;
        }
    }
    let mut tree = Node::build(&t, (0, n - 1)).unwrap();
    let mut res: Vec<i32> = Vec::new();
    
    for i in 0..m {
        if queries[i][0] == 1 {
            let left: usize = queries[i][1] as usize;
            let right: usize = queries[i][2] as usize;
            let mut sum: i32 = tree.query((left, right));
            if left != right {
                sum -= (t[left] + t[right]);
            } else {
                sum -= t[left];
            res.push(sum);
        } else {
            let index: usize = queries[i][1] as usize;
            let value: i32 = queries[i][2];
            let mut temp: i32 = 0;
            
            if index == 0 {
                if arr[index + 1] > value && arr[index + 1] > arr[index + 2] {
                    temp += 1;
                }
                if temp != t[index + 1] {
                    tree.update(index + 1, temp);
                    t[index + 1] = temp;
                }
            } else if index == n - 1 {
                if arr[index - 1] > value && arr[index - 1] > arr[index - 2] {
                    temp += 1;
                }
                if temp != t[index - 1] {
                    tree.update(index - 1, temp);
                    t[index - 1] = temp;
                }
            } else {
                if index > 1 && arr[index - 2] < arr[index - 1] && arr[index - 1] > value {
                    temp += 1;

                }
                
                if temp != t[index - 1] {
                    tree.update(index - 1, temp);
                    t[index - 1] = temp;
                }
                
                temp = 0;
                if arr[index - 1] < value && value > arr[index + 1] {
                    temp += 1;
                }
                
                if temp != t[index] {
                    tree.update(index, temp);
                    t[index] = temp;
                }
                
                temp = 0;
                if index < n - 2 && value < arr[index + 1] && arr[index + 1] > arr[index + 2] {
                    temp += 1;
                }
                
                if temp != t[index + 1] {
                    tree.update(index + 1, temp);
                    t[index + 1] = temp;
                }
            }
            
            arr[index] = value;
        }
    }
    
    
    

    println!("{:?}", res)
    
}

//Leetcode 3569

use std::cmp;
use std::collections::BTreeSet;

#[derive(Debug)]
struct Node {
    //Range of array represented by node
    range: (usize, usize),
    
    //Maximum corresponding value,
    max_value: i64,
    
    //Keep lazy value for range updates:
    lazy: i64, 
    
    //Left child node
    left: Option<Box<Node>>,
    
    //Right child node
    right: Option<Box<Node>>
}


impl Node {
    fn new(range: (usize, usize), max_value: i64) -> Node {
         return Node {range: range, max_value: max_value, lazy: 0, left: None, right: None }
    }
    
    fn build(arr: &Vec<i64>, range: (usize, usize)) -> Option<Box<Self>> {
        if range.0 > range.1 {
            return None;
        }
        
        let mid = (range.0 + range.1) / 2;
        
        let mut node = Node::new(range, arr[range.0]);
        
        if range.0 < range.1 {
            node.left = Self::build(arr, (range.0, mid));
            node.right = Self::build(arr, (mid + 1, range.1));
            let left_max = node.left.as_ref().unwrap().max_value;
            let right_max = node.right.as_ref().unwrap().max_value;
            node.max_value = cmp::max(left_max, right_max);
        }
        
        return Some(Box::new(node))
    }
    
    fn push(&mut self) {
        let left = self.left.as_mut().unwrap();
        let right = self.right.as_mut().unwrap();
        left.max_value += self.lazy;
        left.lazy += self.lazy;
        right.max_value += self.lazy;
        right.lazy += self.lazy;
        self.lazy = 0;
    }
    
    fn query(&mut self, range: (usize, usize)) -> i64 {
        if  range.0 > range.1 {
            return i64::MIN
        }
        
        if  self.range.0 == range.0 && self.range.1 == range.1 {
            return self.max_value
        }
        
        self.push();
        
        let mut max_value = i64::MIN;

        if let Some(ref mut left) = self.left {
            let left_query = left.query((range.0, cmp::min(range.1, left.range.1))); 
            max_value = cmp::max(max_value, left_query)
        }
        
        
        if let Some(ref mut right) = self.right {
            let right_query = right.query((cmp::max(range.0, right.range.0), range.1));
            max_value = cmp::max(max_value, right_query)
        }
        
       return max_value
    }
    
    fn update(&mut self, range: (usize, usize), add: i64) {
        if range.0 > range.1 {
            return 
        }
        
         if self.range.0 == range.0 && self.range.1 == range.1 {
            self.max_value += add;
            self.lazy += add;
        } else {
            self.push();
            let mid = (self.range.0 + self.range.1)/2;
            self.left.as_mut().unwrap().update((range.0, cmp::min(mid, range.1)), add);
            self.right.as_mut().unwrap().update((cmp::max(range.0, mid + 1), range.1), add);
            let left_max = self.left.as_ref().unwrap().max_value;
            let right_max = self.right.as_ref().unwrap().max_value;
            self.max_value = cmp::max(left_max, right_max);
        }
    }
}


fn main() {
    let mut arr: Vec<i32> = vec![2, 1, 4];
    //vec![5, 6, 8, 4, 7, 2, 7];
    let queries: Vec<Vec<i32>> = vec![vec![0,1]];
    let n = arr.len();
    let m = queries.len();
    let mut max_val: usize = 0;
    
    for i in 0..n {
        max_val = cmp::max(max_val, arr[i] as usize);
    }
    
    for i in 0..m {
        max_val = cmp::max(max_val, queries[i][1] as usize);
    }
    
    let mut sieve_array: Vec<bool> = vec![true; max_val + 1];
    let mut count_map: Vec<usize> = vec![0; max_val + 1];
    let mut count: usize = 0;
    sieve_array[0] = false;
    sieve_array[1] = false;
 
    for i in 2..=max_val {
        if sieve_array[i] {
            count_map[i] = count;
            count += 1;
            for j in (2*i..max_val + 1).step_by(i) {
                sieve_array[j] = false;
            }
        }
    }
    
    let mut index_arr: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); count];
    let mut t: Vec<i64> = vec![0; n - 1];
    let mut pre: i64 = 0;
    let mut suff: i64 = 0;
    let mut p: Vec<bool> = vec![false; count];
    let mut s: Vec<bool> = vec![false; count];
    
    for i in 0..n - 1 {
        if sieve_array[arr[i] as usize] {
            if !p[count_map[arr[i] as usize]] {
                pre += 1;
                p[count_map[arr[i] as usize]] = true;
            }
            index_arr[count_map[arr[i] as usize]].insert(i);
        }
        t[i] += pre;
        
        if sieve_array[arr[n - 1 - i] as usize] {
            if !s[count_map[arr[n - 1 - i] as usize]] {
                suff += 1;
                s[count_map[arr[n - 1 - i] as usize]] = true;
            }
        }
        
        t[n - 2 - i] += suff;
    }
    
    if sieve_array[arr[n - 1] as usize] {
        index_arr[count_map[arr[n - 1] as usize]].insert(n - 1);
    }
    
    let mut tree = Node::build(&t, (0, n - 2)).unwrap();
    let mut res: Vec<i64> = Vec::new();
    
    for j in 0..m {
        if sieve_array[arr[queries[j][0] as usize] as usize] {
            let i: usize = *index_arr[count_map[arr[queries[j][0] as usize] as usize]].first().unwrap();
            let k: usize = *index_arr[count_map[arr[queries[j][0] as usize] as usize]].last().unwrap();
            if i == queries[i][0] as usize {
                if k == queries[i][0] as usize {
                    tree.update((0, n - 2), -1);
                } else {
                    let l = *index_arr[count_map[arr[queries[j][0] as usize] as usize]].range(i..).next().unwrap();
                    tree.update((i, l - 1), -1);
                }
            } else {
                if k == queries[i][0] as usize {
                    let l = *index_arr[count_map[arr[queries[j][0] as usize] as usize]].range(..k).next_back().unwrap();
                    tree.update((l, queries[j][0] as usize - 1), - 1);
                }
            }
            index_arr[count_map[arr[queries[j][0] as usize] as usize]].remove(&(queries[j][0] as usize));
        }
        
        if sieve_array[queries[j][1] as usize] {
            if index_arr[count_map[queries[j][1] as usize]].is_empty() {
                tree.update((0, n - 2), 1);
            } else {
                let i: usize = *index_arr[count_map[queries[j][1] as usize]].first().unwrap();
                let k: usize = *index_arr[count_map[queries[j][1] as usize]].last().unwrap();
                if (queries[j][0] as usize) < i {
                    tree.update((0, i - 1), 1);
                } else if (queries[j][0] as usize) > k {
                    tree.update((k, queries[j][0] as usize - 1), 1);
                }
            }
            index_arr[count_map[queries[j][1] as usize]].insert(queries[j][0] as usize);
            arr[queries[j][0] as usize] = queries[j][1];
        }
        
        res.push(tree.query((0, n - 2)));
    }
    
    
     println!("{:?}", res)
    
}
