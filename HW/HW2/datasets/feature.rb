require 'fselector'
puts "\n>============> #{File.basename(__FILE__)}"
r = FSelector::LVF.new

r.data_from_random(100, 3, 10, 5, false)
print(r)
