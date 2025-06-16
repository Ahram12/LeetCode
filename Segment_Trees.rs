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
