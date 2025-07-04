//Leetcode 3504

use std::cmp;

fn longest_palindrome(v: &Vec<char>) -> Vec<usize> {
    let n: usize = v.len();
    let mut f: Vec<Vec<bool>> = vec![vec![false; n]; n];
    let mut l: Vec<usize> = vec![1; n];
    
    for j in 0..n {
        f[j][j] = true;
        for i in (0..j).rev() {
            if v[i] == v[j] {
                if i + 1 < j {
                    f[i][j] = f[i + 1][j - 1];
                } else {
                    f[i][j] = true;
                }
            }
            
            if f[i][j] {
                l[i] = cmp::max(l[i], j - i + 1);
            }
        }
    }
    
    return l
}

fn main() {
    let s: String = "b".to_string();
    let t: String = "aaaa".to_string();
    
    let mut w: Vec<char> = s.chars().collect();
    let m: usize = w.len();
    let mut v: Vec<char> = t.chars().collect();
    let n: usize = v.len();
    v = v.into_iter().rev().collect();
    
    let x: Vec<usize> = longest_palindrome(&w);
    let y: Vec<usize> = longest_palindrome(&v);

    let mut dp: Vec<Vec<usize>>= vec![vec![0; n]; m];
    let mut max_length: usize = 0;
    max_length = cmp::max(*x.iter().max().unwrap(), *y.iter().max().unwrap());
    
    for i in (0..m).rev() {
        for j in (0..n).rev() {
            if w[i] == v[j] {
                if i == m - 1 {
                    if j == n - 1 {
                        dp[i][j] = 2;
                    } else {
                        dp[i][j] = 2 + y[j + 1];
                    }
                } else {
                    if j == n - 1 {
                        dp[i][j] = 2 + x[i + 1];
                    } else {
                        dp[i][j] = 2 + cmp::max(dp[i + 1][j + 1], cmp::max(x[i + 1], y[j + 1]));
                    }
                }
            }
            max_length = cmp::max(max_length, dp[i][j]);
        }
    }
    
    println!("{:?}", max_length)
    
}


//Leetcode 3519
use std::cmp;
use std::collections::VecDeque;

fn convert(num: &VecDeque<usize>, b: &usize) -> VecDeque<usize> {
    let b: usize = *b;
    let mut num: VecDeque<usize> = num.clone();
    let mut base: VecDeque<usize> = VecDeque::new();
   
    loop {
        if num.is_empty() {
            break
        } else {
            let mut quotient: VecDeque<usize> = VecDeque::new();
            let mut sum: usize = 0;
            while !num.is_empty() {
                sum *= 10;
                sum += num.pop_front().unwrap();
                quotient.push_back(sum/b);
                let rem = sum - (sum/b)*b;
                sum -= (sum/b)*b;
                if num.is_empty() {
                    base.push_front(sum);
                }
            }
            while !quotient.is_empty() && quotient[0] == 0 {
                quotient.pop_front();
            }
            num = quotient.clone();
        }
    }

    return base
}

fn helper(b: &usize, num: &VecDeque<usize>, c: &Vec<Vec<i64>>, p: &Vec<Vec<i64>>) -> (i64, bool) {
    const MOD: i64 = 10_i64.pow(9) + 7;
    let b: usize = *b;
    let size = num.len();
    let mut res: i64 = (c[size + b - 2][b - 1] - 1 + p[size][num[0]]) % MOD;
    let mut i: usize = 1;
    while i < size && num[i - 1] <= num[i] {
        res += p[size - i][num[i]] - p[size - i][num[i - 1]];
        res %= MOD;
        i += 1;
    }
    
    if i == size {
        res += 1;
    }

    return (res, i == size)

}

impl Solution {
    pub fn count_numbers(l: String, r: String, b: i32) -> i32 {
        let l: &[u8] = l.as_bytes();
        let r: &[u8] = r.as_bytes();

        let mut lower: VecDeque<usize> = VecDeque::new();
        for i in 0..l.len() {
            lower.push_back((l[i] - b'0') as usize);
        }

        let mut upper: VecDeque<usize> = VecDeque::new();
        for i in 0..r.len() {
            upper.push_back((r[i] - b'0') as usize);
        }

        let b: usize = b as usize;
        let first = convert(&lower, &b).clone();
        let second = convert(&upper, &b).clone();

        let m: usize = cmp::max(first.len(), second.len());
        const MOD: i64 = 10_i64.pow(9) + 7;
    
        let mut c: Vec<Vec<i64>> = vec![vec![0; m + 11]; m + 11];
        for j in 0..m + 11 {
            for i in 0..=j {
                if i == 0 || i == j {
                    c[j][i] = 1;
                } else {
                    c[j][i] = (c[j - 1][i] + c[j - 1][i - 1]) % MOD;
                }
            }
        }

        let mut p: Vec<Vec<i64>> = vec![vec![0; b + 1]; m + 1];
        for j in 1..=m {
            for i in 1..b {
                p[j][i + 1] = (p[j][i] + c[j + b - i - 2][b - i - 1]) % MOD;
            }
        }

        let (res1, sgn) = helper(&b, &first, &c, &p);
        let (res2, _ ) = helper(&b, &second, &c, &p);

        let mut res: i64 = res2 - res1;
        if sgn {
            res += 1;
        }
        
        while res < 0 {
            res += MOD;
        }
        return (res % MOD) as i32

    }
}


//Leetcode 3509
use std::cmp;
use std::collections::BTreeSet;

impl Solution {
    pub fn max_product(nums: Vec<i32>, k: i32, limit: i32) -> i32 {
        let n = nums.len();
        let m: usize = if n & 1 == 0 {12*n + 3} else {12*n + 15};
        let l: i32 = if n & 1 == 0 {6*n as i32} else {6*n as i32 + 6};

        if n & 1 == 0 {
            if k > l || k < -l {
                return -1
            }
        } else {
            if k > l || k < -l + 12 {
                return -1
            }
        }

        let mut odd_map: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); m];
        let mut even_map: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); m];
        let mut res: i32 = -1;
        
        let mut f: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; m]; 2]; n];
        let mut zeros: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; m]; 2]; n];
        
        f[0][1][(nums[0] + l) as usize] = true;

        if nums[0] == 0 {
            zeros[0][1][(nums[0] + l) as usize] = true;
        }
        if nums[0] <= limit {
            odd_map[(nums[0] + l) as usize].insert(nums[0]);
        }
        if k == nums[0] && k <= limit {
            res = cmp::max(res, nums[0]);
        }
        
        for i in 1..n {
            let mut temp_odd: Vec<BTreeSet<i32>> = odd_map.clone();
            let mut temp_even: Vec<BTreeSet<i32>> = even_map.clone();
            for t in 0..2 {
                if t == 0 {
                    for s in -l..=cmp::min(l, l + 2 - nums[i]) {
                        f[i][0][(s + l) as usize] = f[i - 1][0][(s + l) as usize] || f[i - 1][1][(s + l + nums[i]) as usize];
                        if nums[i] == 0 {
                            zeros[i][0][(s + l) as usize] = f[i][0][(s + l) as usize];
                            } else {
                            zeros[i][0][(s + l) as usize]  = zeros[i - 1][0][(s + l) as usize];
                        }
                        if f[i - 1][1][(s + l + nums[i]) as usize] {
                            let upper: i32 = if nums[i] > 0 {limit/nums[i]} else {i32::MAX};
                            let set_iter = odd_map[(s + l + nums[i]) as usize].iter();
                            for &val in set_iter {
                                if val > upper {
                                    break
                                } else {
                                    temp_even[(s + l) as usize].insert(val*nums[i]);
                                    if k == s {
                                        res = cmp::max(res, val*nums[i]);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for s in cmp::max(-l, nums[i] - l)..=l {
                        f[i][1][(s + l) as usize] =  f[i - 1][1][(s + l) as usize] || f[i - 1][0][(s + l - nums[i]) as usize];
                        if nums[i] == 0 {
                            zeros[i][1][(s + l) as usize] = f[i][1][(s + l) as usize];
                        } else {
                            zeros[i][1][(s + l) as usize] = zeros[i - 1][1][(s + l) as usize];
                        }
                        if f[i - 1][0][(s + l - nums[i]) as usize] {
                            let upper: i32 = if nums[i] > 0 {limit/nums[i]} else {i32::MAX};
                            let set_iter = even_map[(s + l - nums[i]) as usize].iter();
                            for &val in set_iter {
                                if val > upper {
                                    break
                                } else {
                                    temp_odd[(s + l) as usize].insert(val*nums[i]);
                                    if k == s {
                                        res = cmp::max(res, val*nums[i]);
                                    }
                                }
                            }
                        }
                    }
                    f[i][1][(nums[i] + l) as usize] = true;
                    if nums[0] == 0 {
                        zeros[i][1][(nums[i] + l) as usize] = true;
                    }
                    if nums[i] <= limit {
                        temp_odd[(nums[i] + l) as usize].insert(nums[i]);
                    }
                    if k == nums[i] && k <= limit {
                        res = cmp::max(res, nums[i]);
                    }
                    
                }
            }
            odd_map = temp_odd.clone();
            even_map = temp_even.clone();
        }
        
        if res > - 1 {
            return res
        } else {
            if !f[n - 1][0][(k + l) as usize] && !f[n - 1][1][(k + l) as usize] {
                return res              
            } else {
                if zeros[n - 1][0][(k + l) as usize] || zeros[n - 1][1][(k + l) as usize] {
                    return 0
                } else {       
                    return -1
                }
            }
        }
    }
}

//Leetcode 3490
#[derive(Debug, Clone)]
struct Element {
    arr: [i64; 10],
    curr_idx: i64,
    len: i64
}

fn main() {
    let d: i64 = 9;
    let mut stk: Vec<Element> = Vec::new();
    stk.push(Element{arr: [0; 10], curr_idx: -1, len: 0});
    let mut ans: Vec<Element> = Vec::new();
   
    while !stk.is_empty() {
        let curr = stk.pop().unwrap();
        if curr.curr_idx < 9 {
            for i in 0..=d-curr.len {
                let mut temp = curr.clone();
                temp.arr[(temp.curr_idx + 1) as usize] = i;
                temp.len += i;
                temp.curr_idx += 1;
                if temp.len == d {
                    ans.push(temp);
                } else {
                    stk.push(temp);
                }
            }
        }
    }
    
    let num: Vec<usize> = vec![9, 9, 9, 3, 8, 7, 5, 1, 1];
    let mut f: Vec<i64> = vec![1; d as usize + 1];
    
    for i in 0..d as usize {
        f[i + 1] = (i as i64 + 1)*f[i];
    }
   
    let mut count: i64 = 0;
    for element in &ans {
        let mut sum: i64 = 0;
        let mut prod: i64 = 1;
        let mut fact: i64 = 1;
        for i in 0..10 {
            sum += (i as i64)*element.arr[i];
            if element.arr[i] > 0 {
                prod *= (i as i64).pow(element.arr[i] as u32);
                fact *= f[element.arr[i] as usize];
            }
        }
        if sum > 0 && prod % sum == 0 {
            let mut temp: Element = element.clone();
            for j in 0..num.len() {
                if j > 0 {
                    for i in 0..num[j] {
                        if temp.arr[i] > 0 {
                            count += f[d as usize - j - 1]/(fact/temp.arr[i]);
                        }
                    }
                } else {
                    for i in 1..num[j] {
                        if temp.arr[i] > 0 {
                            count += f[d as usize - j - 1]/(fact/temp.arr[i]);
                        }
                    }
                }
                
                if temp.arr[num[j]] == 0 {
                    break
                }else {
                    fact /= temp.arr[num[j]];
                    temp.arr[num[j]] -= 1;
                }
            }
        }
    }
   
   
    println!("{:?}", count)
}


//Leetcode 446
use std::collections::HashMap;

impl Solution {
    pub fn number_of_arithmetic_slices(nums: Vec<i32>) -> i32 {
        let size = nums.len();
        let mut dp: Vec<HashMap<i64, i32>> = vec![HashMap::new(); size];
        let mut count: i32 = 0;
    
        for j in 1..size {
            for i in 0..j  {
                let diff: i64 = nums[j] as i64 - nums[i] as i64;
                let val = *dp[i].get(&diff).unwrap_or(&0);
                *dp[j].entry(diff).or_insert(0) += val + 1;
                count += val;
            }
        }
    
        return count 
    }
}

//Leetcode 689
impl Solution {
    pub fn max_sum_of_three_subarrays(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let size = nums.len();
        let mut prefix: Vec<i64> = vec![0; size + 1];
    
        for i in 0..size {
            prefix[i + 1] = prefix[i] + nums[i] as i64;
        }
    
        let mut max1: Vec<usize> = vec![0; size];
        max1[k as usize - 1] = k as usize - 1;
        let mut max2: Vec<usize> = vec![0; size];
        max2[2*k as usize - 1] = 2*k as usize - 1;
        let mut val1: i64 = prefix[2*k as usize];
        let mut val2: i64 = prefix[3*k as usize];
        let mut ret: Vec<i32> = vec![0, k, 2*k];
    
        for i in k as usize..size {
            if prefix[i + 1] - prefix[i + 1 - k as usize] > prefix[max1[i - 1] + 1] - prefix[max1[i - 1] + 1 - k as usize] {
                max1[i] = i;
            } else {
                max1[i] = max1[i - 1];
            }
        
            if i > 2*k as usize - 1 {
                if prefix[i + 1] - prefix[i + 1 - k as usize] + prefix[max1[i - k as usize] + 1] - prefix[max1[i - k as usize] + 1 - k as usize] > val1 {
                    val1 = prefix[i + 1] - prefix[i + 1 - k as usize] + prefix[max1[i - k as usize] + 1] - prefix[max1[i - k as usize] + 1 - k as usize];
                    max2[i] = i;
                } else {
                    max2[i] = max2[i - 1];
                }
            }
        
            if i > 3*k as usize - 1 {
                if prefix[i + 1] - prefix[i + 1 - k as usize] + prefix[max2[i - k as usize] + 1] - prefix[max2[i - k as usize] + 1 - k as usize] + prefix[max1[max2[i - k as usize] - k as usize] + 1] - prefix[max1[max2[i - k as usize] - k as usize] + 1 - k as usize] > val2 {
                    val2 = prefix[i + 1] - prefix[i + 1 - k as usize] + prefix[max2[i - k as usize] + 1] - prefix[max2[i - k as usize] + 1 - k as usize] + prefix[max1[max2[i - k as usize] - k as usize] + 1] - prefix[max1[max2[i - k as usize] - k as usize] + 1 - k as usize];
                    ret[0] = (max1[max2[i - k as usize] - k as usize] + 1 - k as usize) as i32;
                    ret[1] = (max2[i - k as usize] + 1 - k as usize) as i32;
                    ret[2] = (i + 1 - k as usize) as i32;
            }
        }
    }
    
    return ret
    }
}

//Leetcode 790
impl Solution {
    pub fn num_tilings(n: i32) -> i32 {
        if n == 1 || n == 2{
            return n
        } else {
            const MOD: i32 = 10_i32.pow(9) + 7;
            let mut dp: Vec<i32> = vec![1;n as usize + 1];
            dp[2] = 2;
            for i in 3..n as usize + 1 {
                let a = 2*dp[i - 1] % MOD;
                let b = dp[i - 3] % MOD;
                dp[i] = (a + b) % MOD;
            }

            return dp[n as usize] % MOD
        }
    }
}

//Leetcode 1186
use std::cmp;

impl Solution {
    pub fn maximum_sum(arr: Vec<i32>) -> i32 {
        let size = arr.len();
        let mut forward: Vec<i64> = vec![0; size + 1];
        let mut backward: Vec<i64> = vec![0; size + 1];
        let mut pre_min: Vec<i64> = vec![0; size + 1];
        let mut suff_min: Vec<i64> = vec![0; size + 1];
        let mut pre_index: usize = 0;
        let mut suff_index: usize = size;
        let mut min: i64 = 0;
    
        let mut neg_list: Vec<usize> = Vec::new();
        let mut ans: i64 = i64::MIN;
    
        for i in 0..size {
            forward[i + 1] = forward[i] + arr[i] as i64;
            backward[size - 1 - i] =  backward[size - i] + arr[size - 1 - i] as i64;
        
            ans = cmp::max(ans, forward[i + 1] - min);
            min = cmp::min(min, forward[i + 1]);
            if forward[i + 1] > forward[pre_index] {
                pre_index = i + 1;
            } 
            if backward[size - 1 - i] > backward[suff_index] {
                suff_index = size - 1 - i;
            }
            pre_min[i + 1] = cmp::min(pre_min[i], forward[i + 1]);
            suff_min[size - 1 - i] = cmp::min(suff_min[size - i], backward[size - 1 - i]);
        
            if arr[i] < 0 {
                neg_list.push(i)
            }
        }
    
        for index in neg_list {
            if index == 0  {
                if pre_index != 0 {
                    ans = cmp::max(ans, forward[pre_index] - arr[index] as i64);
                }
            } else if index == size - 1 {
                if suff_index != size {
                    ans = cmp::max(ans, backward[index + 1] - arr[index] as i64);
                }
            } else {
                ans = cmp::max(ans, forward[index] - pre_min[index - 1] + (backward[index + 1] - suff_min[index + 1]));
            }
        }
    
        return ans as i32
    }
}

//Leetcode 2830
use std::cmp;
use std::collections::HashMap;

impl Solution {
    pub fn maximize_the_profit(n: i32, mut offers: Vec<Vec<i32>>) -> i32 {
        let mut map: HashMap<i32, Vec<i32>> = HashMap::new();
        let mut gold: HashMap<(i32, i32), i32> = HashMap::new();

        offers.sort_unstable_by_key(|v| v[2]);

        for idx in 0..offers.len() {
            map.entry(offers[idx][1]).and_modify(|v| v.push(offers[idx][0])).or_insert(vec![offers[idx][0]]);
            gold.insert((offers[idx][0], offers[idx][1]), offers[idx][2]);
        }
    
        let mut dp: Vec<i32> = vec![0; n as usize];
    
        for idx in 0..n {
            if map.contains_key(&idx) {
                let vec: Vec<i32> = map.get(&idx).unwrap().clone();
                for i in 0..vec.len() {
                    if vec[i] == 0 {
                        if idx == 0 {
                            dp[idx as usize] = cmp::max(dp[idx as usize], *gold.get(&(vec[i], idx)).unwrap());
                        } else {
                            dp[idx as usize] = cmp::max(dp[idx as usize - 1], cmp::max(dp[idx as usize], *gold.get(&(vec[i], idx)).unwrap()));
                        }
                    } else {
                        dp[idx as usize] = cmp::max(dp[idx as usize - 1], cmp::max(dp[idx as usize], dp[vec[i] as usize - 1] + *gold.get(&(vec[i], idx)).unwrap()));
                    }
                }
            } else {
                if idx > 0 {
                    dp[idx as usize] = dp[idx as usize - 1];
                }
            }
        }
    
        return dp[n as usize - 1]
    }
}

//Leetcode 3082
impl Solution {
    pub fn sum_of_power(mut nums: Vec<i32>, k: i32) -> i32 {
        const MOD: i64 = 10_i64.pow(9) + 7;
        let size = nums.len();
        let mut power: Vec<i64> = vec![1; size];
        let mut dp: Vec<Vec<i64>> = vec![vec![0; k as usize + 1]; size];
    
        for i in 1..size {
            power[i] = power[i - 1] * 2;
            power[i] %= MOD;
        }
     
        nums.sort_unstable();
    
        for i in 0..size {
            for j in 1..k as usize + 1 {
                if i == 0 {
                    if nums[i] == j as i32 {
                        dp[i][j] = 1;
                    }
                } else {
                    dp[i][j] = 2*dp[i - 1][j];
                    dp[i][j] %= MOD;
                    if nums[i] < j as i32 {
                        dp[i][j] += dp[i - 1][j - nums[i] as usize];
                    } else if nums[i] == j as i32 {
                        dp[i][j] += power[i];
                    }
                    dp[i][j] %= MOD;
                }
            }
        }
    
        let mut res = dp[size - 1][k as usize];
        while res < 0 {
            res += MOD;
        }
    
        return (res % MOD) as i32
    }
}
