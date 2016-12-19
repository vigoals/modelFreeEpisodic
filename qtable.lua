if not mfc then
	require "initenv"
end

local qtable = torch.class('QTable')

function qtable:__init (args)
	self.k = args.k or 1
	self.n_actions = args.n_actions or 1
	self.buf = KDTree({maxNum=100})
end

function qtable:knn (representation)
	-- if #self.buf == 0 then
	-- 	return nil
	-- end
	--
	-- local list = InsertSortList(self.k)
	-- for i = 1, #self.buf do
	-- 	local dist = self.buf[i].representation:clone():csub(representation)
	-- 	dist = dist:pow(2):sum()
	-- 	dist = torch.sqrt(dist)
	--
	-- 	local data = {dist=dist, q=self.buf[i].q}
	-- 	list:insert(data)
	-- end
	--
	-- local buf = list.buf
	-- local n = #buf
	-- local q = buf[1].q:clone()
	--
	-- for i = 2, n do
	-- 	q:add(buf[i].q)
	-- end
	-- q:div(n)
	--
	-- return q

	local tmp = self.buf:kNearest(representation, self.k)
	--print(tmp)
	if #tmp == 0 then return nil end
	local q = tmp[1].value:clone()
	for i = 2, #tmp do
		q:add(tmp[i].value)
	end
	q:div(#tmp)
	return q
end

function qtable:q (representation)
	-- for i = 1, #self.buf do
	-- 	if self.buf[i].representation:equal(representation) then
	-- 		return self.buf[i].q
	-- 	end
	-- end

	local tmp = self.buf:find(representation)
	if tmp then
		return tmp.value:clone()
	end

	return self:knn(representation)
end

function qtable:update (representation, action_index, q)
	-- body...
	--print("Q table update")
	-- for i = 1, #self.buf do
	-- 	if self.buf[i].representation:equal(representation) then
	-- 		self.buf[i].q[action_index] =
	-- 			math.max(self.buf[i].q[action_index], q)
	-- 		return
	-- 	end
	-- end
	--
	-- local q_tmp = torch.zeros(self.n_actions)
	-- q_tmp[action_index] = q
	-- local tmp = {representation=representation, q=q_tmp}
	-- self.buf[#self.buf+1] = tmp

	-- if q ~= 0 then
	-- 	print(q)
	-- end

	local tmp = self.buf:find(representation)
	if tmp then
		tmp.value[action_index] = math.max(tmp.value[action_index], q)
	else
		local q_tmp = torch.zeros(self.n_actions)
		q_tmp[action_index] = q
		self.buf:add(representation, q_tmp)
	end
end

function qtable:report (n)
	if #self.buf == 0 then
		print("Q table is empty.")
		return
	end

	print("Q table info")
	for i = 1, n do
		local k = torch.random(#self.buf)
		print(self.buf[k].q)
	end
	print("--------------------")
end
