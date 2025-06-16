//Leetcode 1606
use std::cmp;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;

#[derive(Eq, PartialEq, Debug)]
struct Element {
    index: usize,
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
    pub fn busiest_servers(k: i32, arrival: Vec<i32>, load: Vec<i32>) -> Vec<i32> {
        let mut count: Vec<i32> = vec![0; k as usize]; 
        let mut map: BTreeMap<usize, i32> = BTreeMap::new();
        let mut heap: BinaryHeap<Element> = BinaryHeap::new();
    
        for i in 0..k as usize {
            map.insert(i ,1);
        }
    
        let size = arrival.len();
        let mut max_count: i32 = 0;
        for idx in 0..size {
            while !heap.is_empty() && heap.peek().unwrap().value <= arrival[idx] {
                map.insert(heap.peek().unwrap().index, 1);
                heap.pop();
            }
            if let Some((&key, _)) = map.range((idx % (k as usize))..).next() {
                map.remove(&key);
                heap.push(Element{index: key, value: arrival[idx] + load[idx]});
                count[key] += 1;
                max_count = cmp::max(max_count, count[key]);
            } else {
                if let Some((&key, _)) = map.range(0..).next() {
                    map.remove(&key);
                    heap.push(Element{index: key, value: arrival[idx] + load[idx]});
                    count[key] += 1;
                    max_count = cmp::max(max_count, count[key]);
                }
            }
        }
    
        let mut max_indices: Vec<i32> = Vec::new();
        for i in 0..k as usize {
            if count[i] == max_count {
                max_indices.push(i as i32)
            }
        }
    
        return max_indices
    }
}

//Leetcode 2382
use std::collections::HashMap;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;

impl Solution {
    pub fn maximum_segment_sum(nums: Vec<i32>, remove_queries: Vec<i32>) -> Vec<i64> {
        let size = nums.len();
        let mut prefix: Vec<i64> = vec![0; size + 1];
    
        for i in 0..size {
            prefix[i + 1] = prefix[i] + nums[i] as i64;
        }
    
        let mut ans: Vec<i64> = vec![0; size];
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        let mut hash: HashMap<i64, i64> = HashMap::new();
        let mut heap: BinaryHeap<i64> = BinaryHeap::new();
    
        for j in 0..size {
            if let Some((i, _)) = map.range(..remove_queries[j]).next_back() {
                if let Some((k, _)) = map.range(remove_queries[j]..).next() {
                    hash.entry(prefix[*k as usize] - prefix[*i as usize + 1]).and_modify(|counter| *counter -= 1);
                    if *hash.get(&(prefix[*k as usize] - prefix[*i as usize + 1])).unwrap() == 0 {
                        hash.remove(&(prefix[*k as usize] - prefix[*i as usize + 1]));
                    } 
                    hash.entry(prefix[remove_queries[j] as usize] - prefix[*i as usize + 1]).and_modify(|counter| *counter += 1).or_insert(1);
                    hash.entry(prefix[*k as usize] - prefix[remove_queries[j] as usize + 1]).and_modify(|counter| *counter += 1).or_insert(1);
                    heap.push(prefix[remove_queries[j] as usize] - prefix[*i as usize + 1]);
                    heap.push(prefix[*k as usize] - prefix[remove_queries[j] as usize + 1]);
                } else {
                    hash.entry(prefix[size] - prefix[*i as usize + 1]).and_modify(|counter| *counter -= 1);
                    if *hash.get(&(prefix[size] - prefix[*i as usize + 1])).unwrap() == 0 {
                        hash.remove(&(prefix[size] - prefix[*i as usize + 1]));
                    } 
                    hash.entry(prefix[remove_queries[j] as usize] - prefix[*i as usize + 1]).and_modify(|counter| *counter += 1).or_insert(1);
                    hash.entry(prefix[size] - prefix[remove_queries[j] as usize + 1]).and_modify(|counter| *counter += 1).or_insert(1);
                    heap.push(prefix[remove_queries[j] as usize] - prefix[*i as usize + 1]);
                    heap.push(prefix[size] - prefix[remove_queries[j] as usize + 1]);
                }
            } else {
                if let Some((k, _)) = map.range(remove_queries[j]..).next() {
                    hash.entry(prefix[*k as usize]).and_modify(|counter| *counter -= 1);
                    if *hash.get(&(prefix[*k as usize])).unwrap() == 0 {
                        hash.remove(&(prefix[*k as usize]));
                    } 
                    hash.entry(prefix[remove_queries[j] as usize]).and_modify(|counter| *counter += 1).or_insert(1);
                    hash.entry(prefix[*k as usize] - prefix[remove_queries[j] as usize + 1]).and_modify(|counter| *counter += 1).or_insert(1);
                    heap.push(prefix[remove_queries[j] as usize]);
                    heap.push(prefix[*k as usize] - prefix[remove_queries[j] as usize + 1]);
                } else {
                    hash.entry(prefix[remove_queries[j] as usize]).and_modify(|counter| *counter += 1).or_insert(1);
                    hash.entry(prefix[size] - prefix[remove_queries[j] as usize + 1]).and_modify(|counter| *counter += 1).or_insert(1);
                    heap.push(prefix[remove_queries[j] as usize]);
                    heap.push(prefix[size] - prefix[remove_queries[j] as usize + 1]);
                }
            }
        
            while !hash.contains_key(heap.peek().unwrap()) {
                heap.pop();
            }
            ans[j] = *heap.peek().unwrap();
            map.insert(remove_queries[j], 1);
        
        }
    
        return ans
    }
}

//Leetcode 2763
use std::collections::BTreeMap;

impl Solution {
    pub fn sum_imbalance_numbers(nums: Vec<i32>) -> i32 {
        let size = nums.len();
        let mut total_count: i32 = 0;
    
        for i in 0..size {
            let mut map: BTreeMap<i32, i32> = BTreeMap::new();
            map.insert(nums[i], 1);
            let mut rolling_count: i32 = 0;
            let mut curr_count: i32 = 0;
            for j in i + 1..size {
                if let Some((lower, _)) = map.range(..nums[j]).next_back() {
                    if let Some((upper, _)) = map.range(nums[j]..).next() {
                        if upper - lower > 1 {
                            curr_count -= 1;
                        }
                        if nums[j] - lower > 1 && upper - nums[j] > 1 {
                            curr_count += 2;
                        } else if nums[j] - lower > 1 && upper - nums[j] <= 1{
                            curr_count += 1;
                        } else if nums[j] - lower <= 1 && upper - nums[j] > 1 {
                            curr_count += 1;
                        }
                    } else {
                        if nums[j] - lower > 1 {
                            curr_count += 1;
                        }   
                    }
                } else {
                    if let Some((upper, _)) = map.range(nums[j]..).next() {
                        if upper - nums[j] > 1 {
                            curr_count += 1;
                        }
                    }
                }
                rolling_count += curr_count;
                map.insert(nums[j], 1);
            }
            total_count += rolling_count;
        }
    
        return total_count
    }
}
