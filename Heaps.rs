//Leetcode 3321
use std::cmp;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Element {
    count: i64,
    value: i64,
}

impl Ord for Element {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count)
            .then_with(|| self.value.cmp(&other.value))
    }
}

impl PartialOrd for Element {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct Balance {
    active: BinaryHeap<Element>,
    inactive: BinaryHeap<Element>,
    is_active: Vec<bool>,
    len: i32,
    sum: i64,
}

impl Balance {
    fn add(&mut self, value: &i64, count: &i64, map: &Vec<i64>, x: &i32) {
        if !self.is_active[*value as usize] {
            self.is_active[*value as usize] = true;
            self.len += 1;
            self.sum += count*map[*value as usize];
        } else {
            self.sum += map[*value as usize];
        }
        self.active.push(Element{count: -1**count, value: -1*value});
        
        if self.len > *x {
            let top_count = -self.active.peek().unwrap().count;
            let top_value = -self.active.peek().unwrap().value;
            
            self.active.pop();
            self.inactive.push(Element{count: top_count, value: top_value});
            self.is_active[top_value as usize] = false;
            self.len -= 1;
            self.sum -= top_count*map[top_value as usize];
        }
    }
    
    fn subtract(&mut self, value: &i64, count: &i64, map: &Vec<i64>, x: &i32) {
        if !self.is_active[*value as usize] {
            self.inactive.push(Element{count: *count, value: *value});
        } else {
            self.active.push(Element{count: -1**count, value: -1*value});
            self.sum -= map[*value as usize];
            
            if !self.inactive.is_empty() {
                let top_count = self.inactive.peek().unwrap().count;
                let top_value = self.inactive.peek().unwrap().value;
             
                self.active.push(Element{count: -top_count, value: -top_value});
                self.inactive.pop();
                self.is_active[top_value as usize] = true;
                self.len += 1;
                self.sum += top_count*map[top_value as usize];
                
                if self.len > *x {
                    let top_count = -self.active.peek().unwrap().count;
                    let top_value = -self.active.peek().unwrap().value;
                    self.active.pop();
                    self.inactive.push(Element{count: top_count, value: top_value});
                    self.is_active[top_value as usize] = false;
                    self.len -= 1;
                    self.sum -= top_count*map[top_value as usize];
                }
            }
        }
    }
    
    fn clean(&mut self, counts: &Vec<i64>) {
        while !self.active.is_empty() {
            let top_count = -self.active.peek().unwrap().count;
            let top_value = -self.active.peek().unwrap().value;
            
            if !self.is_active[top_value as usize] || top_count != counts[top_value as usize] {
                self.active.pop();
            } else {
                break
            }
        }
        
        while !self.inactive.is_empty() {
            let top_count = self.inactive.peek().unwrap().count;
            let top_value = self.inactive.peek().unwrap().value;
            
            if self.is_active[top_value as usize] || top_count != counts[top_value as usize] {
                self.inactive.pop();
            } else {
                break
            }
        }
    }
}

        

//, 3, 4, 4, 8, 8, 8, 8
fn main() { 
    let mut nums: Vec<i32> = vec![4,6,5,3,7,7,7,6,8,2,2, 3, 4, 4, 8, 8, 8, 8];
    let k: usize = 5;
    let x: i32 = 3;
    let size = nums.len();
    let mut temp: Vec<i32> = nums.clone();
    temp.sort_unstable();
    let mut i: usize = 0;
    let mut j: usize = 0;
    let mut num: i64 = 0;
    let mut map: Vec<i64> = vec![0; size];
    let mut hash: HashMap<i32, i32> = HashMap::new();
    
    while i < size {
        map[num as usize] = temp[i] as i64;
        hash.insert(temp[i], num as i32);
        while j < size && temp[i] == temp[j] {
            j += 1;
        }
        num += 1;
        i = j;
    }
    
    for i in 0..size {
        nums[i] = *hash.get(&nums[i]).unwrap();
    }
    
    let mut balance: Balance = Balance{active: BinaryHeap::new(), inactive: BinaryHeap::new(), is_active: vec![false; size], len: 0, sum: 0};
    let mut counts: Vec<i64> = vec![0; size];
    let mut res: Vec<i64> = Vec::new();
    for i in 0..size {
        counts[nums[i] as usize] += 1;
        balance.add(&(nums[i] as i64), &counts[nums[i] as usize], &map, &x);
        balance.clean(&counts);
        if i == size - 1 {
            //println!("{:?}", balance);
        }
        if i == k - 1 {
            res.push(balance.sum);
        }
        if i >= k  {
            counts[nums[i - k] as usize] -= 1;
            balance.subtract(&(nums[i - k] as i64), &counts[nums[i - k] as usize], &map, &x);
            balance.clean(&counts); 
            res.push(balance.sum);
        }
        
    }
    
    println!("{:?}", res)
}

//Leetcode 3691
use std::cmp;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Element {
    start: usize,
    end: usize,
    value: usize,
}

impl Ord for Element {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
            .then_with(|| self.end.cmp(&other.end))
    }
}

impl PartialOrd for Element {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn main() {
    let mut nums: Vec<i32> = Vec::new();
    let len: i32 = 50_000_i32;
    for i in 0..len {
        nums.push(i);
    }
    let k: usize = 2;
    let size = nums.len();
    let mut max_range: Vec<Vec<usize>> = vec![vec![0, 0]; size];
    let mut min_range: Vec<Vec<usize>> = vec![vec![0, 0]; size];
    let mut max_stk_front: Vec<usize> = Vec::new();
    let mut max_stk_back: Vec<usize> = Vec::new();
    let mut min_stk_front: Vec<usize> = Vec::new();
    let mut min_stk_back: Vec<usize> = Vec::new();
    let mut zip: Vec<Vec<usize>> = vec![vec![0, 0]; size];
    
    for i in 0..size {
        zip[i][0] = nums[i] as usize;
        zip[i][1] = i;
        
        while !max_stk_front.is_empty() && nums[max_stk_front[max_stk_front.len() - 1]] <= nums[i] {
            max_stk_front.pop();
        }
        if max_stk_front.is_empty() {
            max_range[i][0] = 0;
        } else {
            max_range[i][0] = max_stk_front[max_stk_front.len() - 1] + 1;
        }
        max_stk_front.push(i);
        
        while !min_stk_front.is_empty() && nums[min_stk_front[min_stk_front.len() - 1]] >= nums[i] {
            min_stk_front.pop();
        }
        if min_stk_front.is_empty() {
            min_range[i][0] = 0;
        } else {
            min_range[i][0] = min_stk_front[min_stk_front.len() - 1] + 1;
        }
        min_stk_front.push(i);
        
        while !max_stk_back.is_empty() && nums[max_stk_back[max_stk_back.len() - 1]] < nums[size - 1 - i] {
            max_stk_back.pop();
        }
        if max_stk_back.is_empty() {
            max_range[size - 1 - i][1] = size - 1;
        } else {
            max_range[size - 1 - i][1] = max_stk_back[max_stk_back.len() - 1] - 1;
        }
        max_stk_back.push(size - 1 - i);
        
        while !min_stk_back.is_empty() && nums[min_stk_back[min_stk_back.len() - 1]] > nums[size - 1 - i] {
            min_stk_back.pop();
        }
        if min_stk_back.is_empty() {
            min_range[size - 1 - i][1] = size - 1;
        } else {
            min_range[size - 1 - i][1] = min_stk_back[min_stk_back.len() - 1] - 1;
        }
        min_stk_back.push(size - 1 - i);
    }
    
    zip.sort_by_key(|v| (v[0], v[1]));
    
    let mut num_arrs: usize = 0;
    let mut j: usize = size - 1;
    let mut i = 0;
    let mut pq: BinaryHeap<Element> = BinaryHeap::new();
    let mut next: i32 = size as i32 - 2;
    let mut distances: usize = 0;

    while num_arrs < k {
        if j < i {
            if pq.is_empty() {
                if next < 0 {
                    break
                } else {
                    i = 0;
                    j = next as usize;
                    next -= 1;
                }
            } else {
                let prev: Element = pq.pop().unwrap();
                i = prev.start;
                j = prev.end;
            }
        }
        
        let mut curr_dist = zip[j][0] - zip[i][0];
        
        if !pq.is_empty() && curr_dist < pq.peek().unwrap().value {
            pq.push(Element{start: i, end: j, value: curr_dist});
            if next > -1 && pq.peek().unwrap().value < zip[next as usize][0] - zip[0][0] {
                i = 0;
                j = next as usize;
                next -= 1;
            } else {
                let prev: Element = pq.pop().unwrap();
                i = prev.start;
                j = prev.end;
            }
        } else {
            if next > - 1 && curr_dist < zip[next as usize][0] - zip[0][0] {
                pq.push(Element{start: i, end: j, value: curr_dist});
                i = 0;
                j = next as usize;
                next -= 1;
            }
        }

        curr_dist = zip[j][0] - zip[i][0];
        let min: usize = zip[i][1];
        let min_lower = min_range[min][0];
        let min_upper = min_range[min][1];
        
        let max: usize = zip[j][1];
        let max_lower = max_range[max][0];
        let max_upper = max_range[max][1];
        
        if (max >= min_lower && max <= min_upper) && (min >= max_lower && min <= max_upper) {
            let lower = cmp::max(min_lower, max_lower);
            let upper = cmp::min(min_upper, max_upper);
            let left = cmp::min(min, max);
            let right = cmp::max(min, max);
            let new_sets = (left - lower + 1)*(upper - right + 1);
            
            if num_arrs + new_sets >= k {
                distances += curr_dist * (k - num_arrs);
                num_arrs = k;
            } else {
                distances += curr_dist*new_sets;
                num_arrs += new_sets;
                
            }
        } 
        i += 1;
    }
        
    
    println!("{:?}",distances);
}

//Leetcode 857
use std::collections::BinaryHeap;

impl Solution {
    pub fn mincost_to_hire_workers(quality: Vec<i32>, wage: Vec<i32>, k: i32) -> f64 {
        let size = quality.len();
        let mut zip: Vec<Vec<i32>> = vec![vec![0];size];
        for idx in 0..size {
            zip[idx] = vec![quality[idx], wage[idx]];
        }
        zip.sort_by(|a, b| (&((a[1] as f64)/(a[0] as f64))).partial_cmp((&((b[1] as f64)/(b[0] as f64)))).unwrap());
    
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        let mut sum: i32 = 0;
        let mut res: f64 = f64::MAX;
    
        for idx in 0..size {
            sum += zip[idx][0];
            heap.push(zip[idx][0]);
            let scale: f64 = (zip[idx][1] as f64)/(zip[idx][0] as f64);
            if heap.len() > k as usize {
                sum -= heap.pop().unwrap();
            }
            if heap.len() == k as usize {
                let temp: f64 =  scale*(sum as f64);
                if temp < res {
                    res = temp;
                }
            }
        }
    
        return res
    }
}

//Leetcode 1354
use std::cmp;
use std::collections::BinaryHeap;

impl Solution {
    pub fn is_possible(target: Vec<i32>) -> bool {
        let size = target.len();
    
        if size == 1 {
            return target[0] == 1
        }
    
        let mut heap: BinaryHeap<i64> = BinaryHeap::new();
        let mut sum: i64 = 0;
    
        for i in 0..size {
            sum += target[i] as i64;
            heap.push(target[i] as i64);
        }
    
        while *heap.peek().unwrap() > 1 {
            if 2*(*heap.peek().unwrap()) <= sum {
                return false
            } else {
                let mut curr_max = heap.pop().unwrap();
                sum -= curr_max;
                let k = cmp::max((curr_max - *heap.peek().unwrap())/(sum), 1);
                curr_max -= sum*k;
                if curr_max!= 1 && curr_max == *heap.peek().unwrap() {
                    return false
                }
                sum += curr_max;
                heap.push(curr_max);
            }
                if *heap.peek().unwrap() == 1 {
                    return true
                }
        }
    
        return true
    }
}

//Leetcode 1499
use std::cmp;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Element {
    position: i32,
    value: i32,
}

impl Ord for Element {
    fn cmp(&self, other: &Self) -> Ordering {
        other.value.cmp(&self.value)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for Element {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Solution {
    pub fn find_max_value_of_equation(points: Vec<Vec<i32>>, k: i32) -> i32 {
        let size = points.len();
        let mut heap: BinaryHeap<Element> = BinaryHeap::new();
        let mut max_val: i32 = i32::MIN;
    
        for i in 0..size {
            if !heap.is_empty() {
                while !heap.is_empty() {
                    if points[i][0] - heap.peek().unwrap().position > k {
                        heap.pop();
                    } else {
                        max_val = cmp::max(points[i][0] + points[i][1] - heap.peek().unwrap().value, max_val);
                        heap.push(Element{position: points[i][0], value: points[i][0] - points[i][1]}); 
                        break
                    }
                }
                if heap.is_empty() {
                    heap.push(Element{position: points[i][0], value: points[i][0] - points[i][1]});
                }
            } else {
                heap.push(Element{position: points[i][0], value: points[i][0] - points[i][1]});
            }
        }
    
        return max_val
    }
}

//Leetocde 1851
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Eq, PartialEq, Debug)]
struct Element {
    index: i32,
    value: i32,
}

impl Ord for Element {
    fn cmp(&self, other: &Self) -> Ordering {
        other.value.cmp(&self.value)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Element {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Solution {
    pub fn min_interval(mut intervals: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
        let mut update_quer: Vec<Vec<i32>> = Vec::new();
        for (idx, query) in queries.iter().enumerate() {
            update_quer.push(vec![*query, idx as i32]);
        }
    
        update_quer.sort_unstable_by_key(|v| v[0]);
        intervals.sort_unstable_by_key(|v| v[0]);
    
        let mut res: Vec<i32> = vec![-1; update_quer.len()];
        let mut heap = BinaryHeap::new();
        let mut prev_idx: usize = 0;
        let mut curr_idx: usize = 0;
    
        let mut start: usize = 0;
        while intervals[0][0] > update_quer[start][0] {
            start += 1;
        }
    
        for idx in start..update_quer.len() {
            let mut low: usize = prev_idx;
            let mut high: usize = intervals.len() - 1;
        
            while low <= high {
                let mid = (low + high)/2;
                if intervals[mid][0] <= update_quer[idx][0] {
                    if mid == intervals.len() - 1 {
                        curr_idx = intervals.len() - 1;
                        break
                    } else {
                        if intervals[mid + 1][0] <= update_quer[idx][0] {
                            low = mid + 1;
                        } else {
                            curr_idx = mid;
                            break
                        }
                    }
                } else {
                    high = mid - 1;
                }
            }
            
            if curr_idx >= prev_idx {
                for i in prev_idx..curr_idx + 1 {
                    heap.push(Element{index: intervals[i][1], value: intervals[i][1] - intervals[i][0] + 1});
                }
            }
            
            while !heap.is_empty() && heap.peek().unwrap().index < update_quer[idx][0] {
                heap.pop();
            }
            
            if !heap.is_empty() {
                res[update_quer[idx][1] as usize] = heap.peek().unwrap().value;
            }
            prev_idx = curr_idx + 1; 
        }
    
    
        return res
    }
}

//Leetocde 3013
use std::cmp;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Element {
    value: i32,
    position: usize,
}

impl Ord for Element {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
            .then_with(|| other.position.cmp(&self.position))
    }
}


impl PartialOrd for Element {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Solution {
    pub fn minimum_cost(nums: Vec<i32>, k: i32, dist: i32) -> i64 {
        let size = nums.len();
        let mut curr_sum: i64 = nums[0] as i64;
        let mut in_heap = BinaryHeap::new();
        let mut out_heap = BinaryHeap::new();
        let mut map: HashMap<usize, usize> = HashMap::new();
    
        let start: usize = 1;
        let mut end: usize = 1;
    
        while in_heap.len() < k as usize - 1 && end - start <= dist as usize {
            in_heap.push(Element{value: nums[end], position: end});
            map.insert(end, end);
            curr_sum += nums[end] as i64;
            end += 1;
        }
    
        let mut opt_sum = curr_sum;
    
    
        while end - start <= dist as usize && end < size {
            let prev_index = in_heap.peek().unwrap().position;
            let prev_value = in_heap.peek().unwrap().value;
            if prev_value >= nums[end] {
                map.remove(&prev_index);
                in_heap.pop();
                out_heap.push(Element{value: -prev_value, position: prev_index});
                curr_sum -= prev_value as i64;
            
                in_heap.push(Element{value: nums[end], position: end});
                curr_sum += nums[end] as i64;
                map.insert(end, end);
                opt_sum = cmp::min(opt_sum, curr_sum);
            } else {
                out_heap.push(Element{value: -nums[end], position: end});
            }
            end += 1;
        }

        for start in 2.. size - (dist as usize) {
            if map.contains_key(&(start - 1)) {
                if in_heap.peek().unwrap().position as usize == start - 1 {
                    in_heap.pop();
                }
                curr_sum -= nums[start - 1] as i64;
                map.remove(&(start - 1));
                while !out_heap.is_empty() && out_heap.peek().unwrap().position < start {
                    out_heap.pop();
                }
                if out_heap.is_empty() {
                    in_heap.push(Element{value: nums[end], position: end});
                    curr_sum += nums[end] as i64;
                    map.insert(end, end);
                    opt_sum = cmp::min(opt_sum, curr_sum);
                } else {
                    let prev_index = out_heap.peek().unwrap().position;
                    let prev_value = -out_heap.peek().unwrap().value;
                
                    if prev_value < nums[end] {
                        out_heap.pop();
                        in_heap.push(Element{value: prev_value, position: prev_index});
                        curr_sum += prev_value as i64;
                        map.insert(prev_index, prev_index);
                        opt_sum = cmp::min(opt_sum, curr_sum);
                        out_heap.push(Element{value: -nums[end], position: end});
                    } else {
                        in_heap.push(Element{value: nums[end], position: end});
                        curr_sum += nums[end] as i64;
                        map.insert(end, end);
                        opt_sum = cmp::min(opt_sum, curr_sum);
                    }
                }
            } else {
                while !in_heap.is_empty() && in_heap.peek().unwrap().position < start {
                    in_heap.pop();
                }
                let prev_index = in_heap.peek().unwrap().position;
                let prev_value = in_heap.peek().unwrap().value;
                if prev_value >= nums[end] {
                    if map.contains_key(&prev_index) {
                        map.remove(&prev_index);
                    }
                    if prev_index > start {
                        out_heap.push(Element{value: -prev_value, position: prev_index});
                    }
                    curr_sum -= prev_value as i64;
                    in_heap.pop();
                    in_heap.push(Element{value: nums[end], position: end});
                    curr_sum += nums[end] as i64;
                    map.insert(end, end);
                    opt_sum = cmp::min(opt_sum, curr_sum);
                } else {
                    out_heap.push(Element{value: -nums[end], position: end});
                }
            }
            end += 1;
        }
    
        return opt_sum
    }
}
