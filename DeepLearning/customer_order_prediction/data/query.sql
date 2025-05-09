with last_order_date as 
(
select max(order_date) as max_date from orders
),
customer_order_stats as (
select 
c.customer_id,
count(o.order_id) as total_orders,
sum(od.unit_price * od.quantity) as total_spent,
avg(od.unit_price * od.quantity) as avg_order_value,
max(o.order_date) as last_order_date
from
customers c inner join orders o
on c.customer_id=o.customer_id
inner join order_details od
on o.order_id = od.order_id
group by c.customer_id),

label_data as(
select c.customer_id,
case
when exists (
select 1 from orders o2,last_order_date lod
	where o2.customer_id=c.customer_id
	and o2.order_date> (lod.max_date - Interval '6 months')
)
then 1 else 0
end as will_order_again
from customers c
)
select 
s.customer_id,
s.total_orders,
s.total_spent,
s.avg_order_value,
s.last_order_date,
l.will_order_again
from customer_order_stats s join label_data l
on s.customer_id=l.customer_id